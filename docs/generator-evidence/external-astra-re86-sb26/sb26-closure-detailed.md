# SB26 closure detailed evidence

See [sb26-closure-summary.md](sb26-closure-summary.md) for closure of all3 prior findings, the remaining P2 provenance binding, and retained limitations.

# SB26 closure review evidence

## Initial hashes

- `pebby/games/sb26/__init__.py`: `c1b526e9ddb4287df3234d4088bece1db58ae9300ccdb5357aaec6101ca1e062`
- `pebby/games/sb26/bank.py`: `d4928ff25e19d83c6a5f344f83f5fca447b8478cdb10434d958ca68b3f4cb88e`
- `pebby/games/sb26/env.py`: `0c0052f6605d15f32fd5aafd82523cc8d43b160ef01322162fa023a3a18209f3`
- `pebby/games/sb26/generate.py`: `fce3556486bf4852c9e33102a7e4584052dc3db3a2f23d45c44b0104761b5fd8`
- `pebby/games/sb26/layout.py`: `71f8826863335b1f9e184089161cf346d0deb6bce8c103adb59c43121ac78e67`
- `pebby/games/sb26/names.py`: `dccc1226cf70b4de95e2991ca350cbbd4a8f5730f992b3f13d15fdf32cdb1e6c`
- `pebby/games/sb26/plan.py`: `0b8d789d93ea20289c4dfb8b1cd42ea4846c101a3f6745fefc0cd5d65c851901`
- `pebby/games/sb26/reference_profiles.py`: `6e446ada4dbe562fc496baa33ecf0232dfb4525fc4328c8e323f6f2d093f358a`
- `tests/games/test_sb26.py`: `c3769ae0dadd9837516eccd3908fd55b25be29959ed2faeb9904c23875463c79`
- `tests/games/test_sb26_quality.py`: `f5ed0d1282be5d09ff76b4865aa22b9ca2b45b4d9f2a821eea9ed5c069de77e9`

## Independent closure/native probes

```json
{
  "full_builds": {
    "enriched": 8,
    "standalone": 8
  },
  "native_replays": [
    {
      "difficulty": 1,
      "won": true,
      "completed": 1,
      "first_completion_index": 8,
      "actions": 9,
      "energy": [
        64,
        64,
        63,
        63,
        62,
        62,
        61,
        61,
        60,
        59
      ]
    },
    {
      "difficulty": 2,
      "won": true,
      "completed": 1,
      "first_completion_index": 14,
      "actions": 15,
      "energy": [
        64,
        64,
        63,
        63,
        62,
        62,
        61,
        61,
        60,
        60,
        59,
        59,
        58,
        58,
        57,
        56
      ]
    },
    {
      "difficulty": 3,
      "won": true,
      "completed": 1,
      "first_completion_index": 14,
      "actions": 15,
      "energy": [
        64,
        64,
        63,
        63,
        62,
        62,
        61,
        61,
        60,
        60,
        59,
        59,
        58,
        58,
        57,
        56
      ]
    },
    {
      "difficulty": 4,
      "won": true,
      "completed": 1,
      "first_completion_index": 14,
      "actions": 15,
      "energy": [
        64,
        64,
        63,
        63,
        62,
        62,
        61,
        61,
        60,
        60,
        59,
        59,
        58,
        58,
        57,
        56
      ]
    },
    {
      "difficulty": 5,
      "won": true,
      "completed": 1,
      "first_completion_index": 16,
      "actions": 17,
      "energy": [
        64,
        64,
        63,
        63,
        62,
        62,
        61,
        61,
        60,
        60,
        59,
        59,
        58,
        58,
        57,
        57,
        56,
        55
      ]
    },
    {
      "difficulty": 6,
      "won": true,
      "completed": 1,
      "first_completion_index": 18,
      "actions": 19,
      "energy": [
        64,
        64,
        63,
        63,
        62,
        62,
        61,
        61,
        60,
        60,
        59,
        59,
        58,
        58,
        57,
        57,
        56,
        56,
        55,
        54
      ]
    },
    {
      "difficulty": 7,
      "won": true,
      "completed": 1,
      "first_completion_index": 16,
      "actions": 17,
      "energy": [
        64,
        64,
        63,
        63,
        62,
        62,
        61,
        61,
        60,
        60,
        59,
        59,
        58,
        58,
        57,
        57,
        56,
        55
      ]
    },
    {
      "difficulty": 8,
      "won": true,
      "completed": 1,
      "first_completion_index": 16,
      "actions": 17,
      "energy": [
        64,
        64,
        63,
        63,
        62,
        62,
        61,
        61,
        60,
        60,
        59,
        59,
        58,
        58,
        57,
        57,
        56,
        55
      ]
    }
  ],
  "strict_mutations": [
    {
      "path": [
        "proof",
        "engine_win"
      ],
      "errors": [
        "engine_win=1, expected exact True"
      ]
    },
    {
      "path": [
        "proof",
        "levels_completed"
      ],
      "errors": [
        "levels_completed=True, expected exact 1"
      ]
    },
    {
      "path": [
        "context_index"
      ],
      "errors": [
        "context_index=False, expected exact 0"
      ]
    },
    {
      "path": [
        "native_budget"
      ],
      "errors": [
        "native_budget=64.0, expected exact 64"
      ]
    },
    {
      "path": [
        "engine_win"
      ],
      "errors": [
        "engine_win=False, expected exact True"
      ]
    },
    {
      "path": [
        "teacher_model_exact"
      ],
      "errors": [
        "teacher_model_exact=False, expected exact True"
      ]
    },
    {
      "path": [
        "search_truncated"
      ],
      "errors": [
        "search_truncated=True, expected exact False"
      ]
    },
    {
      "path": [
        "search_unsupported"
      ],
      "errors": [
        "search_unsupported=True, expected exact False"
      ]
    },
    {
      "path": [
        "solution_energy_cost"
      ],
      "errors": [
        "solution_energy_cost=999, expected exact 5"
      ]
    },
    {
      "path": [
        "generation_limits",
        "action_limit"
      ],
      "errors": [
        "action_limit must be at least 9"
      ]
    },
    {
      "path": [
        "generation_exclusions"
      ],
      "errors": [
        "bounded generation rejection counts are missing or malformed"
      ]
    },
    {
      "path": [
        "proof",
        "search_limit"
      ],
      "errors": [
        "search_limit=500000.0, expected exact 500000"
      ]
    },
    {
      "path": [
        "context_solution",
        0,
        0
      ],
      "errors": [
        "context_solution must exactly mirror solution"
      ]
    },
    {
      "path": [
        "generation_limits",
        "attempts"
      ],
      "errors": [
        "attempts cannot exceed 10000"
      ]
    },
    {
      "path": [
        "generation_limits",
        "node_limit"
      ],
      "errors": [
        "node_limit cannot exceed 1000000"
      ]
    },
    {
      "path": [
        "proof",
        "first_completion_action_index"
      ],
      "errors": [
        "first_completion_action_index=0, expected exact 8"
      ]
    },
    {
      "path": [
        "solution_mechanics",
        "route_minimum_energy"
      ],
      "errors": [
        "stored mechanic-use certificate does not match route replay"
      ]
    },
    {
      "path": [
        "format"
      ],
      "errors": [
        "format='pebby.sb26.level.v2', expected exact 'pebby.sb26.level.v3'"
      ]
    },
    {
      "path": [
        "geometry_version"
      ],
      "errors": [
        "geometry_version='logical-translation-colour-v1', expected exact 'native-role-semantics-v2'"
      ]
    }
  ],
  "missing_proof_field_rejections": 39,
  "post_win_suffix_errors": [
    "generated witness must contain click pairs followed by one submit"
  ],
  "relative_frame_variants": [
    {
      "shift": -3,
      "identities": {
        "geometry_sha256": "0884fbf5bb06444453fd0946987630b2198c43eb1327c83fe39e9a1d63acc232",
        "geometry_d4_sha256": "0884fbf5bb06444453fd0946987630b2198c43eb1327c83fe39e9a1d63acc232",
        "geometry_split": "validation",
        "gameplay_sha256": "80949bb663465230f0643886f85ae0d80c102dd4a219010e92bb09ffff3b0b8c",
        "raw_start_frame_sha256": "b2dd125f3f9b177f8b13efa01ed6cab628c4c36e8965029427f12b6b3481c9b0"
      },
      "native": {
        "won": true,
        "completed": 1,
        "first_completion_index": 14,
        "actions": 15,
        "energy": [
          64,
          64,
          63,
          63,
          62,
          62,
          61,
          61,
          60,
          60,
          59,
          59,
          58,
          58,
          57,
          56
        ]
      }
    },
    {
      "shift": 0,
      "identities": {
        "geometry_sha256": "0884fbf5bb06444453fd0946987630b2198c43eb1327c83fe39e9a1d63acc232",
        "geometry_d4_sha256": "0884fbf5bb06444453fd0946987630b2198c43eb1327c83fe39e9a1d63acc232",
        "geometry_split": "validation",
        "gameplay_sha256": "80949bb663465230f0643886f85ae0d80c102dd4a219010e92bb09ffff3b0b8c",
        "raw_start_frame_sha256": "862c2b62f4eb33a42cae4c23d030f03ae4d83eff23541875c617b90c2dd0ffd3"
      },
      "native": {
        "won": true,
        "completed": 1,
        "first_completion_index": 14,
        "actions": 15,
        "energy": [
          64,
          64,
          63,
          63,
          62,
          62,
          61,
          61,
          60,
          60,
          59,
          59,
          58,
          58,
          57,
          56
        ]
      }
    },
    {
      "shift": 3,
      "identities": {
        "geometry_sha256": "0884fbf5bb06444453fd0946987630b2198c43eb1327c83fe39e9a1d63acc232",
        "geometry_d4_sha256": "0884fbf5bb06444453fd0946987630b2198c43eb1327c83fe39e9a1d63acc232",
        "geometry_split": "validation",
        "gameplay_sha256": "80949bb663465230f0643886f85ae0d80c102dd4a219010e92bb09ffff3b0b8c",
        "raw_start_frame_sha256": "ba9b36eadb5c95957c3ba4cd3ce2ae051d8d7d7a3c292c563347c2af413c875a"
      },
      "native": {
        "won": true,
        "completed": 1,
        "first_completion_index": 14,
        "actions": 15,
        "energy": [
          64,
          64,
          63,
          63,
          62,
          62,
          61,
          61,
          60,
          60,
          59,
          59,
          58,
          58,
          57,
          56
        ]
      }
    }
  ],
  "border_collision": [
    {
      "ids": {
        "geometry_sha256": "76d86f104091c1a23bd13e182e334648634a01b7ad4ce99736eda846713db1c7",
        "geometry_d4_sha256": "76d86f104091c1a23bd13e182e334648634a01b7ad4ce99736eda846713db1c7",
        "geometry_split": "validation",
        "gameplay_sha256": "4bbf31d82242560d2f8724e9006bd04164ce303f0600d3cfb94c75e20c540e9b",
        "raw_start_frame_sha256": "4a846e23971c1291adcc8140aa150c6a82ffa82a577e5a2c1af8b9be1f2aa3a2"
      },
      "native": {
        "won": true,
        "completed": 1,
        "first_completion_index": 8,
        "actions": 9,
        "energy": [
          64,
          64,
          63,
          63,
          62,
          62,
          61,
          61,
          60,
          59
        ]
      }
    },
    {
      "ids": {
        "geometry_sha256": "76d86f104091c1a23bd13e182e334648634a01b7ad4ce99736eda846713db1c7",
        "geometry_d4_sha256": "76d86f104091c1a23bd13e182e334648634a01b7ad4ce99736eda846713db1c7",
        "geometry_split": "validation",
        "gameplay_sha256": "4bbf31d82242560d2f8724e9006bd04164ce303f0600d3cfb94c75e20c540e9b",
        "raw_start_frame_sha256": "18e9eba408e4cde3f74a0689d62de63eb98679401821f30dc6519331cf2e13ec"
      },
      "native": {
        "won": true,
        "completed": 1,
        "first_completion_index": 8,
        "actions": 9,
        "energy": [
          64,
          64,
          63,
          63,
          62,
          62,
          61,
          61,
          60,
          59
        ]
      }
    }
  ],
  "link_relabel": {
    "ids_unchanged": true,
    "native": {
      "won": true,
      "completed": 1,
      "first_completion_index": 16,
      "actions": 17,
      "energy": [
        64,
        64,
        63,
        63,
        62,
        62,
        61,
        61,
        60,
        60,
        59,
        59,
        58,
        58,
        57,
        57,
        56,
        55
      ]
    }
  },
  "link_retarget": {
    "ids_changed": true,
    "native": {
      "won": false,
      "completed": 0,
      "first_completion_index": null,
      "actions": 17,
      "energy": [
        64,
        64,
        63,
        63,
        62,
        62,
        61,
        61,
        60,
        60,
        59,
        59,
        58,
        58,
        57,
        57,
        56,
        55
      ]
    }
  },
  "natural_pair": [
    {
      "seed": 0,
      "split": "train",
      "order": [
        0,
        3,
        2,
        1
      ],
      "ids": {
        "geometry_sha256": "43f906c46c385bcd8bfe4ab9d7b256d9af908f01b591355792962d21d572335b",
        "geometry_d4_sha256": "43f906c46c385bcd8bfe4ab9d7b256d9af908f01b591355792962d21d572335b",
        "geometry_split": "train",
        "gameplay_sha256": "ba45c81c3e4aad6e82bf55e7c9a1ccafe5520f1d5f3c6b04564239a068e8109e",
        "raw_start_frame_sha256": "0ee67e2159889754994117db1606fa56eac2e2adf10fcefbecd2d98c791ea9da"
      },
      "native": {
        "won": true,
        "completed": 1,
        "first_completion_index": 8,
        "actions": 9,
        "energy": [
          64,
          64,
          63,
          63,
          62,
          62,
          61,
          61,
          60,
          59
        ]
      }
    },
    {
      "seed": 2,
      "split": "test",
      "order": [
        3,
        2,
        0,
        1
      ],
      "ids": {
        "geometry_sha256": "ef5e5dc91f987fc153b9846996e41ef3a4b31b00ba8f8ca2dde1931ea3da0d0e",
        "geometry_d4_sha256": "ef5e5dc91f987fc153b9846996e41ef3a4b31b00ba8f8ca2dde1931ea3da0d0e",
        "geometry_split": "test",
        "gameplay_sha256": "339cbf31cf0a8a5874eee279ebed5aa7f59be4c4ef3761864c6555310a6b12c6",
        "raw_start_frame_sha256": "4c1035c7b93e6415391779073fe6c919509b0730c1867201331392db6759844d"
      },
      "native": {
        "won": true,
        "completed": 1,
        "first_completion_index": 8,
        "actions": 9,
        "energy": [
          64,
          64,
          63,
          63,
          62,
          62,
          61,
          61,
          60,
          59
        ]
      }
    }
  ],
  "relabel_enriched_game_seed": {
    "result": "ACCEPTED",
    "claimed_game_seed": 5678,
    "first_claimed_child_seed": 2280357784076989336,
    "first_actual_requested_seed": 3153776018888810025
  }
}
```

## Exhaustive finite tutorial state check (no official search)

Construct each of the24 goal-to-tray orderings directly, replay native clicks without search, then check admission.

```json
{
  "unique_semantics": 24,
  "grammar_support": {
    "test": 4,
    "validation": 11,
    "train": 9
  },
  "admitted_support": {
    "test": 4,
    "validation": 10,
    "train": 9
  },
  "direct_native_wins": 24,
  "rejected": [
    {
      "goal_order": [
        8,
        11,
        14,
        9
      ],
      "split": "validation",
      "reason": "official_copy",
      "gameplay": "b934f9c69b073bc443aa8baa057e11b66b6acba9a7db3422fc27a541c913c3ed"
    }
  ],
  "bounded_failure": {
    "report": {
      "accepted": false,
      "seed": 41001,
      "effective_seed": 41001,
      "difficulty": 1,
      "split": "train",
      "attempts": 2,
      "rejections": {
        "geometry_split": 2
      }
    },
    "events": [
      {
        "seed": 41001,
        "difficulty": 1,
        "attempt": 1,
        "reason": "geometry_split"
      },
      {
        "seed": 41001,
        "difficulty": 1,
        "attempt": 2,
        "reason": "geometry_split"
      }
    ]
  }
}
```

## Residual enriched-provenance reproduction

```python
from pebby.games.sb26.generate import generate_game, build_game, _child_seed
rows = generate_game(1234, split="validation")
assert rows is not None
for row in rows:
    row["game_seed"] = 5678
    row["child_seed"] = _child_seed(5678, row["difficulty"])
assert rows[0]["child_seed"] != rows[0]["requested_seed"]
assert len(build_game(rows)) == 8  # current defect: accepts false seed provenance
```

The new all-or-none field check and sequence identity check do not close this missing equality. Merely changing the claimed game seed plus its derived child labels leaves real level bytes, all existing single-level proofs, and the sequence hash untouched. Severity P2: provenance integrity/reproducibility, not native playability. Suggested minimal repair is an exact equality guard between each `child_seed` and validated `requested_seed`, with a negative whole-game relabel test.

## Scope and process accounting

Read complete current worker sb26.md/sb26-final.md, external-next-corrections/sb26-final.md, complete detailed prior review, and actual current changed source/tests. The source inventory proved that adapter/planner/layout/profile/names files are unchanged. One heavy process at a time, primary venv with worker PYTHONPATH, PYTHONDONTWRITEBYTECODE, numerical threads1,1900MiB address-space limit and115-second alarm. Sessions20798 and80223 both completed normally with exit0. No spawned children, subagents, servers, compositors, browsers, training or official expensive searches. Full64 audit and official route searches were not rerun.

## Final hashes

All initial/final scoped hashes match. Frozen 2026-09-19 00:45:46 CEST.

- `pebby/games/sb26/__init__.py`: `c1b526e9ddb4287df3234d4088bece1db58ae9300ccdb5357aaec6101ca1e062`
- `pebby/games/sb26/bank.py`: `d4928ff25e19d83c6a5f344f83f5fca447b8478cdb10434d958ca68b3f4cb88e`
- `pebby/games/sb26/env.py`: `0c0052f6605d15f32fd5aafd82523cc8d43b160ef01322162fa023a3a18209f3`
- `pebby/games/sb26/generate.py`: `fce3556486bf4852c9e33102a7e4584052dc3db3a2f23d45c44b0104761b5fd8`
- `pebby/games/sb26/layout.py`: `71f8826863335b1f9e184089161cf346d0deb6bce8c103adb59c43121ac78e67`
- `pebby/games/sb26/names.py`: `dccc1226cf70b4de95e2991ca350cbbd4a8f5730f992b3f13d15fdf32cdb1e6c`
- `pebby/games/sb26/plan.py`: `0b8d789d93ea20289c4dfb8b1cd42ea4846c101a3f6745fefc0cd5d65c851901`
- `pebby/games/sb26/reference_profiles.py`: `6e446ada4dbe562fc496baa33ecf0232dfb4525fc4328c8e323f6f2d093f358a`
- `tests/games/test_sb26.py`: `c3769ae0dadd9837516eccd3908fd55b25be29959ed2faeb9904c23875463c79`
- `tests/games/test_sb26_quality.py`: `f5ed0d1282be5d09ff76b4865aa22b9ca2b45b4d9f2a821eea9ed5c069de77e9`
