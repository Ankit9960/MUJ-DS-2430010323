"""Published MIT-BIH results used for the CO4 comparison.

Every row is inter-patient (de Chazal DS1/DS2) unless its `protocol_note` says
otherwise, and every row carries its source so the table in the report is
citable rather than decorative.

IMPORTANT, and worth a paragraph in the report: the spread in this table is
enormous -- 86% to 99.9% accuracy on nominally the same protocol. That spread is
not progress alone. Several of the high-scoring entries apply oversampling
(e.g. SMOTE) or beat-level augmentation in ways that can place synthetic
neighbours of a test beat into the training set, which reintroduces exactly the
leakage the inter-patient protocol exists to prevent. The conservative reading
is that the classical entries (de Chazal, Zhang) are directly comparable to a
clean re-implementation, while the >99% entries are comparable only if their
resampling was strictly confined to training data.

Verify each number against the cited paper before submission.
"""

SOTA_INTER_PATIENT = [
    dict(
        study="de Chazal et al.",
        year=2004,
        method="Linear discriminant (morphology + RR features)",
        accuracy=86.2,
        N=(87.1, 99.2), S=(76.0, 38.5), V=(80.3, 86.6), F=(89.4, 8.6),
        protocol_note="Defines the DS1/DS2 inter-patient protocol.",
        source="IEEE Trans. Biomed. Eng. 51(7):1196-1206. PMID 15248536",
    ),
    dict(
        study="Zhang et al.",
        year=2014,
        method="Disease-specific feature selection + SVM ensemble",
        accuracy=88.9,
        N=(98.98, 98.98), S=(79.06, 35.98), V=(85.48, 92.75), F=(93.81, 13.73),
        protocol_note="Inter-patient, 4 classes reported.",
        source="Pattern Recognition 47(6):2154-2165",
    ),
    dict(
        study="Acharya et al.",
        year=2014,
        method="Nonlinear features + classifier ensemble",
        accuracy=93.0,
        N=None, S=None, V=None, F=None,
        protocol_note="Per-class Se/P+ not reported in the comparison sources.",
        source="Reported in Mousavi & Afghah (ICASSP 2019), Table 3",
    ),
    dict(
        study="Garcia et al.",
        year=2017,
        method="Temporal vectorcardiogram + SVM",
        accuracy=92.4,
        N=None, S=None, V=None, F=None,
        protocol_note="Per-class Se/P+ not reported in the comparison sources.",
        source="Reported in Mousavi & Afghah (ICASSP 2019), Table 3",
    ),
    dict(
        study="Mousavi & Afghah",
        year=2019,
        method="Sequence-to-sequence RNN with attention",
        accuracy=99.53,
        N=(99.68, 99.55), S=(88.94, 92.57), V=(99.94, 99.50), F=None,
        protocol_note="SMOTE oversampling applied to the training set. "
                      "Treat as an optimistic upper reference.",
        source="ICASSP 2019, arXiv:1812.07421",
    ),
    dict(
        study="Frontiers in Physiology",
        year=2023,
        method="MLP + weight capsule + seq2seq",
        accuracy=99.88,
        N=(99.79, 99.90), S=(97.66, 96.14), V=(99.97, 99.07), F=(97.94, 98.70),
        protocol_note="Heavy resampling; near-ceiling scores on a protocol where "
                      "classical methods reach ~87% invite scrutiny.",
        source="Front. Physiol. 14:1247587",
    ),
]

# Evidence that the protocol itself moves the number, independent of the model.
PROTOCOL_EVIDENCE = dict(
    claim="Identical CNN, evaluated both ways",
    intra_accuracy=99.48,
    inter_accuracy=88.34,
    delta=11.14,
    source="Systematic review, arXiv:2503.07276, reference [57]",
    quote=("Including heartbeats from the same patient in both phases introduces "
           "bias, as models memorize patient-specific characteristics during "
           "training, leading to inflated test performance (often approaching 100%)."),
)


def markdown_table() -> str:
    """Render the SOTA comparison as a markdown table."""
    fmt = lambda p: f"{p[0]:.1f} / {p[1]:.1f}" if p else "—"
    lines = [
        "| Study | Year | Method | Acc (%) | N Se/P+ | S Se/P+ | V Se/P+ | F Se/P+ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in SOTA_INTER_PATIENT:
        lines.append(
            f"| {r['study']} | {r['year']} | {r['method']} | {r['accuracy']:.1f} | "
            f"{fmt(r['N'])} | {fmt(r['S'])} | {fmt(r['V'])} | {fmt(r['F'])} |"
        )
    return "\n".join(lines)


def source_list() -> str:
    return "\n".join(
        f"- **{r['study']} ({r['year']})** — {r['source']}. _{r['protocol_note']}_"
        for r in SOTA_INTER_PATIENT
    )
