# sabr

Can be downloaded with 
```bash
git clone --recurse-submodules https://github.com/delalamo/sabr.git
```

TODO:
- [ ] Get a protoype script up and running for camelid and mouse
- [ ] Verify that scores make sense, and figure out cutoffs for poor alignments (like non-Igs)
- [ ] Identify outliers and remove (MMSeqs2 clusters)
- [ ] Make RFDiffusion mAbs and VHHs with partial diffusion and see if numbering is recovered
- [ ] Make CIF and PDB both parseable
- [ ] Write tests for methods
- [ ] Add options to trim residues <1 and >128
- [ ] Make containers that can be pulled