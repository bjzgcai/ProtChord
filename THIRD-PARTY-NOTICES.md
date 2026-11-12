# Third-party notices

The MIT license in `LICENSE` applies to ProtChord contributions. It does not
replace the licenses or copyright notices of third-party material.

- fairseq (Meta/Facebook): MIT, https://github.com/facebookresearch/fairseq .
  Copyright (c) Facebook, Inc. and its affiliates.
  Portions of `protchord_core/modules/` and encoder attention modules retain
  their original copyright notices.
- Graphormer (Microsoft): MIT, https://github.com/microsoft/Graphormer .
  Copyright (c) Microsoft Corporation.
  Public reference for Graphormer-derived encoder components.
- MACE: MIT, https://github.com/ACEsuit/mace . Copyright (c) 2022
  ACEsuit/mace. Portions of
  `protchord_core/models/encoder/equivariant/maceblocks.py` identify MACE as
  their source.
- EquiformerV2 and fairchem: MIT, https://github.com/atomicarchitects/equiformer_v2
  and https://github.com/FAIR-Chem/fairchem . Copyright (c) 2023 Yi-Lun Liao
  and Copyright (c) Meta, Inc. and its affiliates, respectively. Encoder
  equivariant modules include adapted implementations from these projects.
- torchscale: MIT, https://github.com/microsoft/torchscale . Encoder and
  transformer components include adapted implementations. Copyright (c)
  Microsoft Corporation.
- e3nn: MIT, https://github.com/e3nn/e3nn/tree/0.4.0. The Wigner implementation
  in `protchord_core/models/encoder/equivariant/equiformer_v2/wigner.py` is
  adapted from e3nn 0.4.0 and remains subject to its copyright and MIT license.
  Copyright (c) 2020, The Regents of the University of California, through
  Lawrence Berkeley National Laboratory (subject to receipt of any required
  approvals from the U.S. Dept. of Energy), Ecole Polytechnique Federale de
  Lausanne (EPFL), Free University of Berlin and Kostiantyn Lapchevskyi. All
  rights reserved.
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions: The above copyright
  notice and this permission notice shall be included in all copies or
  substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
  WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
  TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
  FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
  THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Meta Llama 3 tokenizer

`tokenizer/` includes a tokenizer derived from the Llama 3 vocabulary. That
material is subject to the Meta Llama 3 Community License, rather than the
repository MIT license. The agreement and acceptable-use policy are included
in `LICENSE-META-LLAMA-3`; the required attribution is in `NOTICE`.

The Meta license includes redistribution, attribution, naming, acceptable-use,
and commercial-use conditions. Downstream users must comply with those terms.
