# Component-Wise Layer Alignment Metric

For each target surface component:

1. enumerate printable layer surfaces,
2. find the layer index that minimizes RMS height error on that component,
3. report the component RMS under the best match.

This avoids forcing a global one-layer-to-one-surface correspondence, which becomes ambiguous for internal ceilings and terraced roofs.
