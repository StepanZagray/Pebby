"""Compact symbolic state of a TR87 level, read out of a live `Env`.

Everything upstream's win test (`bsqsshqpox`, tr87.py:1044-1105) looks at is a
sprite *name*, a rule's position in the strip-sorted rule list, or a marker
sprite sitting under a rule's first left-hand tile. So a level is fully
described by symbols such as "B6" (family letter + digit 1..7), and the whole
search state is the editable digits plus the cursor. `translate` mirrors the
upstream parse exactly, including its quirks (greedy first rule, `zip`
truncation in double translation, only `lhs[0]` consulted in tree translation,
free trailing target tiles).
"""

from dataclasses import dataclass, field

from . import names


@dataclass(frozen=True)
class Layout:
    source: tuple            # symbols of the source row, left to right
    target: tuple            # symbols of the editable target row (upstream y, x order)
    rules: tuple             # ((lhs symbols), (rhs symbols)) in upstream rule order
    cursor: int
    budget: int              # actions left
    alter_rules: bool = False
    double_translation: bool = False
    tree_translation: bool = False
    links: tuple = ()        # double translation: (primary rule index, secondary rule index)
    secondary: tuple = ()    # double translation: rule indices carrying a "...2" marker

    @property
    def group_count(self):
        """How many groups ACTION3/ACTION4 cycle through (tr87.py:997-1002)."""
        return 2 * len(self.rules) if self.alter_rules else len(self.target)

    def group_symbols(self, index):
        """Symbols of editable group `index` (a target tile, or one rule side)."""
        if self.alter_rules:
            lhs, rhs = self.rules[index // 2]
            return tuple(lhs if index % 2 == 0 else rhs)
        return (self.target[index],)

    def to_dict(self):
        return {"source": list(self.source), "target": list(self.target),
                "rules": [[list(lhs), list(rhs)] for lhs, rhs in self.rules],
                "cursor": self.cursor, "budget": self.budget,
                "alter_rules": self.alter_rules, "double_translation": self.double_translation,
                "tree_translation": self.tree_translation,
                "links": [list(pair) for pair in self.links], "secondary": list(self.secondary)}

    @classmethod
    def from_dict(cls, data):
        return cls(source=tuple(data["source"]), target=tuple(data["target"]),
                   rules=tuple((tuple(lhs), tuple(rhs)) for lhs, rhs in data["rules"]),
                   cursor=data["cursor"], budget=data["budget"],
                   alter_rules=bool(data.get("alter_rules")),
                   double_translation=bool(data.get("double_translation")),
                   tree_translation=bool(data.get("tree_translation")),
                   links=tuple(tuple(pair) for pair in data.get("links", ())),
                   secondary=tuple(data.get("secondary", ())))


def extract(env):
    """Read the symbolic state of `env`'s current level."""
    rule_sprites = env.rule_sprites()
    rules = tuple((tuple(names.symbol(s.name) for s in lhs), tuple(names.symbol(s.name) for s in rhs))
                  for lhs, rhs in rule_sprites)
    links, secondary = [], []
    if env.flag(names.KEY_DOUBLE_TRANSLATION):
        # Mirrors `lonhgifaes` (tr87.py:1045-1053): a "...1" marker under lhs[0]
        # chains this rule to the rule whose lhs[0] sits under the "...2" marker.
        for index, (lhs, _) in enumerate(rule_sprites):
            marker = env.marker_at(lhs[0])
            if marker is None:
                continue
            if marker.endswith("2"):
                secondary.append(index)
                continue
            partners = env.level.get_sprites_by_name(marker.replace("1", "2"))
            if not partners:
                raise ValueError(f"marker {marker} has no partner")
            tile = env.level.get_sprite_at(partners[0].x, partners[0].y, names.TAG_TILE)
            partner = next((j for j, (lhs2, _) in enumerate(rule_sprites) if tile is lhs2[0]), None)
            if partner is None:
                raise ValueError(f"marker {marker} points at no rule")
            links.append((index, partner))
    return Layout(source=tuple(env.source_row()), target=tuple(env.target_row()), rules=rules,
                  cursor=env.cursor(), budget=env.budget_left(),
                  alter_rules=env.flag(names.KEY_ALTER_RULES),
                  double_translation=env.flag(names.KEY_DOUBLE_TRANSLATION),
                  tree_translation=env.flag(names.KEY_TREE_TRANSLATION),
                  links=tuple(links), secondary=tuple(secondary))


def translate(layout, rules=None):
    """The target prefix upstream's win test demands, or None if the parse fails.

    Exact mirror of `bsqsshqpox` with the target comparison factored out: the
    check succeeds iff `translate(...)` is a list `T` and `target[:len(T)] == T`
    (with `len(T) <= len(target)`). `rules` overrides the layout's rules (alter
    mode changes them mid-game).
    """
    rules = layout.rules if rules is None else rules
    links = dict(layout.links)
    secondary = set(layout.secondary)

    def expand(index):
        lhs, rhs = rules[index]
        if index in secondary:
            return None
        if index in links:
            lhs2, rhs2 = rules[links[index]]
            return tuple(lhs) + tuple(lhs2), tuple(rhs) + tuple(rhs2)
        return tuple(lhs), tuple(rhs)

    source = layout.source
    position, out = 0, []
    while position < len(source):
        for index, (lhs, rhs) in enumerate(rules):
            if source[position:position + len(lhs)] != tuple(lhs):
                continue
            if layout.tree_translation:
                mapped, ok = [], True
                for sym in rhs:
                    for lhs2, rhs2 in rules:
                        if lhs2[0] == sym:
                            mapped.extend(rhs2)
                            break
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                rhs = mapped
            elif layout.double_translation:
                expanded = expand(index)
                if expanded is None:
                    continue
                lhs, rhs = expanded
                for other in range(len(rules)):
                    expanded = expand(other)
                    if expanded is None:
                        continue
                    lhs2, rhs2 = expanded
                    if all(a == b for a, b in zip(rhs, lhs2)):
                        rhs = rhs2
                        break
                else:
                    continue
            out.extend(rhs)
            position += len(lhs)
            break
        else:
            return None
    return out


def translation_trace(layout, rules=None):
    """Replay the winning grammar and report the branch/rules actually used."""
    rules = layout.rules if rules is None else rules
    links = dict(layout.links)
    secondary = set(layout.secondary)

    def expand(index):
        lhs, rhs = rules[index]
        if index in secondary:
            return None
        if index in links:
            lhs2, rhs2 = rules[links[index]]
            return tuple(lhs) + tuple(lhs2), tuple(rhs) + tuple(rhs2)
        return tuple(lhs), tuple(rhs)

    branch = "tree" if layout.tree_translation else (
        "double" if layout.double_translation else "direct")
    position, output = 0, []
    outer_matches, secondary_matches = [], []
    tree_expansions = double_compositions = 0
    tree_mixed_child_expansions = tree_repeated_child_expansions = 0
    while position < len(layout.source):
        for index, (lhs, rhs) in enumerate(rules):
            if layout.source[position:position + len(lhs)] != tuple(lhs):
                continue
            consumed = tuple(lhs)
            mapped = []
            if branch == "tree":
                children = tuple(rhs)
                matches = []
                for symbol in children:
                    for other, (lhs2, rhs2) in enumerate(rules):
                        if lhs2[0] == symbol:
                            mapped.extend(rhs2)
                            matches.append(other)
                            break
                    else:
                        break
                else:
                    rhs = tuple(mapped)
                    secondary_matches.extend(matches)
                    tree_expansions += len(matches)
                    if len(children) >= 2:
                        if len(set(children)) == 1:
                            tree_repeated_child_expansions += 1
                        else:
                            tree_mixed_child_expansions += 1
                    outer_matches.append(index)
                    output.extend(rhs)
                    position += len(consumed)
                    break
                continue
            if branch == "double":
                expanded = expand(index)
                if expanded is None:
                    continue
                consumed, rhs = expanded
                for other in range(len(rules)):
                    second = expand(other)
                    if second is None:
                        continue
                    lhs2, rhs2 = second
                    if all(a == b for a, b in zip(rhs, lhs2)):
                        rhs = rhs2
                        secondary_matches.append(other)
                        double_compositions += 1
                        break
                else:
                    continue
            outer_matches.append(index)
            output.extend(rhs)
            position += len(consumed)
            break
        else:
            return {"success": False, "branch": branch, "output": tuple(output),
                    "outer_rule_matches": tuple(outer_matches),
                    "secondary_rule_matches": tuple(secondary_matches),
                    "translation_depth": 0, "tree_expansions": tree_expansions,
                    "tree_mixed_child_expansions": tree_mixed_child_expansions,
                    "tree_repeated_child_expansions": tree_repeated_child_expansions,
                    "double_compositions": double_compositions}
    composed = ((branch == "tree" and tree_expansions > 0)
                or (branch == "double" and double_compositions > 0))
    depth = 2 if composed else 1
    return {"success": True, "branch": branch, "output": tuple(output),
            "outer_rule_matches": tuple(outer_matches),
            "secondary_rule_matches": tuple(secondary_matches),
            "translation_depth": depth, "tree_expansions": tree_expansions,
            "tree_mixed_child_expansions": tree_mixed_child_expansions,
            "tree_repeated_child_expansions": tree_repeated_child_expansions,
            "double_compositions": double_compositions}


def solved(layout, target=None, rules=None):
    """Would upstream's win test pass for this target row and these rules?"""
    target = layout.target if target is None else target
    demanded = translate(layout, rules)
    return demanded is not None and len(demanded) <= len(target) and list(target[:len(demanded)]) == demanded
