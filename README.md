# lbu-metal-safe

`lbu-metal-safe` is the deliberately narrow unsafe boundary used by Little Big Universe's direct
Metal presenter. It turns window-borrowed layer attachment, slice-qualified resource transfers,
and exact drawable presentation callbacks into safe Rust operations.

It is not a renderer abstraction or RHI. Scene meaning, shaders, pipelines, command encoding,
resource policy, and presentation contracts remain outside this crate.
