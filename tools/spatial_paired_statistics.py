"""Small, dependency-free statistics for paired binary level outcomes.

The unit of inference is a frozen level, so wins from the two arms are paired.
The McNemar calculation conditions on discordant levels and uses the exact
binomial null.  The interval is an exact Clopper--Pearson interval for the
candidate share of discordant wins. A separate conservative interval for the
net difference uses simultaneous binomial bounds on gain and loss probabilities;
it includes uncertainty in how often pairs disagree. Ties are excluded only
from the conditional McNemar test, not from those marginal probability bounds.
"""

from math import comb, exp, fsum, lgamma, log, log1p


def _validate_confidence(confidence):
    if isinstance(confidence, bool) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    return float(confidence)


def _binomial_tail(successes, trials, probability, *, lower):
    """Return P(X >= successes) or P(X <= successes) for X~Binomial."""
    if not 0 <= successes <= trials:
        return 0.0 if lower else 0.0
    if probability == 0.0:
        return float(successes == 0) if lower else 1.0
    if probability == 1.0:
        return 1.0 if lower else float(successes == trials)
    if lower:
        indices = range(successes, trials + 1)
    else:
        indices = range(0, successes + 1)
    log_factorial = lgamma(trials + 1)
    log_p, log_q = log(probability), log1p(-probability)
    return min(1.0, fsum(exp(log_factorial - lgamma(k + 1) - lgamma(trials - k + 1)
                            + k * log_p + (trials - k) * log_q) for k in indices))


def exact_binomial_interval(successes, trials, confidence=0.95, iterations=80):
    """Return a two-sided exact Clopper--Pearson interval.

    The inversion is done by bisection over exact finite binomial tails, so no
    SciPy or incomplete-beta implementation is required.  ``trials == 0`` is
    represented by ``(None, None)`` because there is no estimand.
    """
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise ValueError("successes and trials must be integers")
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise ValueError("successes and trials must be integers")
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes must be in 0..trials")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 20:
        raise ValueError("iterations must be an integer of at least 20")
    confidence = _validate_confidence(confidence)
    if trials == 0:
        return None, None
    alpha = (1.0 - confidence) / 2.0
    if successes == 0:
        lower = 0.0
    else:
        left, right = 0.0, 1.0
        for _ in range(iterations):
            mid = (left + right) / 2.0
            if _binomial_tail(successes, trials, mid, lower=True) >= alpha:
                right = mid
            else:
                left = mid
        lower = right
    if successes == trials:
        upper = 1.0
    else:
        left, right = 0.0, 1.0
        for _ in range(iterations):
            mid = (left + right) / 2.0
            if _binomial_tail(successes, trials, mid, lower=False) > alpha:
                left = mid
            else:
                right = mid
        upper = right
    return lower, upper


def exact_mcnemar_two_sided(reference_only_wins, candidate_only_wins):
    """Exact two-sided McNemar p-value from discordant paired outcomes."""
    for value in (reference_only_wins, candidate_only_wins):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("discordant counts must be nonnegative integers")
    discordant = reference_only_wins + candidate_only_wins
    if discordant == 0:
        return 1.0
    lower_tail = sum(comb(discordant, k) for k in range(0, min(reference_only_wins,
                                                               candidate_only_wins) + 1))
    # Divide integers before converting their bounded ratio to float. Casting
    # either enormous count first overflows on otherwise ordinary large panels.
    p_value = (2 * lower_tail) / (2 ** discordant)
    return min(1.0, p_value)


def _rows_by_id(rows, identifier, side):
    result = {}
    for row in rows:
        if not isinstance(row, dict) or identifier not in row:
            raise ValueError(f"{side} rows need identifier {identifier!r}")
        key = row[identifier]
        if key in result:
            raise ValueError(f"duplicate paired identifier {key!r} in {side} rows")
        result[key] = row
    return result


def paired_win_statistics(reference, candidate, *, outcome='completed', identifier=None,
                          confidence=0.95):
    """Summarize paired binary outcomes and exact uncertainty.

    ``reference`` and ``candidate`` may be sequences of booleans or mappings.
    For mappings, ``identifier`` pairs common rows and records unpaired rows;
    this is useful for honest prefix reports after a bounded timeout.  With no
    identifier, equal-length positional pairing is required.
    """
    confidence = _validate_confidence(confidence)
    unpaired_reference = []
    unpaired_candidate = []
    if identifier is None:
        if len(reference) != len(candidate):
            raise ValueError("positional paired outcomes must have equal length")
        pairs = list(zip(reference, candidate))
    else:
        left = _rows_by_id(reference, identifier, 'reference')
        right = _rows_by_id(candidate, identifier, 'candidate')
        common = sorted(left.keys() & right.keys())
        unpaired_reference = sorted(left.keys() - right.keys())
        unpaired_candidate = sorted(right.keys() - left.keys())
        pairs = [(left[key].get(outcome), right[key].get(outcome)) for key in common]
    counts = dict(both_wins=0, both_losses=0, reference_only_wins=0,
                  candidate_only_wins=0)
    for ref, cand in pairs:
        if not isinstance(ref, bool) or not isinstance(cand, bool):
            raise ValueError("paired outcomes must be boolean")
        if ref and cand:
            counts['both_wins'] += 1
        elif not ref and not cand:
            counts['both_losses'] += 1
        elif ref:
            counts['reference_only_wins'] += 1
        else:
            counts['candidate_only_wins'] += 1
    n = len(pairs)
    reference_only = counts['reference_only_wins']
    candidate_only = counts['candidate_only_wins']
    discordant = reference_only + candidate_only
    interval = exact_binomial_interval(candidate_only, discordant, confidence)
    if interval[0] is None:
        conditional_share = None
    else:
        conditional_share = candidate_only / discordant
    if n:
        # Bonferroni: each marginal interval fails with probability at most
        # alpha/2, so both hold with probability >= 1-alpha. Independence
        # between gain and loss counts is NOT assumed.
        marginal_confidence = 1.0 - (1.0 - confidence) / 2.0
        if marginal_confidence == 1.0:
            # Rounding at the largest representable confidence must widen the
            # interval, never silently lower its requested coverage.
            net_interval = (-1.0, 1.0)
        else:
            gain_bounds = exact_binomial_interval(candidate_only, n, marginal_confidence)
            loss_bounds = exact_binomial_interval(reference_only, n, marginal_confidence)
            net_interval = (gain_bounds[0] - loss_bounds[1], gain_bounds[1] - loss_bounds[0])
    else:
        net_interval = (None, None)
    return dict(
        paired_levels=n,
        reference_wins=counts['both_wins'] + reference_only,
        candidate_wins=counts['both_wins'] + candidate_only,
        both_wins=counts['both_wins'],
        both_losses=counts['both_losses'],
        reference_only_wins=reference_only,
        candidate_only_wins=candidate_only,
        net_wins=candidate_only - reference_only,
        net_win_rate_difference=((candidate_only - reference_only) / n if n else None),
        discordant_pairs=discordant,
        conditional_candidate_win_share=conditional_share,
        conditional_candidate_win_share_exact_interval=interval,
        net_win_rate_difference_conservative_interval=net_interval,
        interval_confidence=confidence,
        interval_method='Clopper-Pearson exact binomial on candidate share among discordant pairs',
        net_interval_method='Bonferroni simultaneous Clopper-Pearson bounds on gain and loss probabilities',
        inference_limits='Binomial bounds assume independent identically sampled paired levels; fixed, stratified, reused, or selected panels require design-aware interpretation and are not untouched confirmation.',
        mcnemar_exact_two_sided_p=exact_mcnemar_two_sided(reference_only, candidate_only),
        unpaired_reference=unpaired_reference,
        unpaired_candidate=unpaired_candidate,
    )
