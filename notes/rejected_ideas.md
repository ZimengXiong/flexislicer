# Rejected Ideas

## Spatially varying blend `w(x,y,z)`

Rejected because

- `∇D` gains an extra `(S_high - S_low) ∇w` term
- clean slope inheritance disappears
- manufacturability arguments become local and messy

## Learned basis `S_k`

Rejected because

- weak guarantees on slope and invertibility
- harder to reproduce from paper artifacts alone
- adds data dependence to a method whose strength is closed-form reasoning

## Direct per-layer fields

Rejected because

- too many degrees of freedom
- awkward coupling between layers
- much harder to constrain or explain

## `K=3+` for the first submission

Deferred because

- it expands target extraction and coupling logic significantly
- the core claim only requires demonstrating that `K=2` breaks the `K=1` ceiling
- scope control matters more than saturating capability in version one
