# lbu-metal-safe

`lbu-metal-safe` is the deliberately narrow unsafe boundary used by Little Big Universe's direct
Metal presenter. It turns window-borrowed layer attachment, slice-qualified resource transfers,
checked retained render submission, and exact drawable presentation callbacks into safe Rust
operations.

It is not a renderer abstraction or RHI. Scene meaning, shaders, pipeline policy, render-pass
selection, resource policy, and presentation contracts remain outside this crate.
