# sabr

Can be downloaded with 
```bash
git clone --recurse-submodules https://github.com/delalamo/sabr.git
```

TODO:
- [ ] Get a protoype script up and running for camelid, human, and mouse
    * I need to be systematic about this and align clustered, filtered arrays to each LOO model in a representative way, testing various conditions
    * Conditions to test:
        * Keeping all possible positions
        * Omitting positions without an AHo equivalent
        * Omitting all insert positions
        * Omitting CDRs
- [ ] Implement and test radial basis functions as substitutes for dot products, to permit introduction of a sigma parameter
- [ ] Verify that scores make sense, and figure out cutoffs for poor alignments (like non-Igs)
- [ ] Identify outliers and remove (MMSeqs2 clusters)
- [ ] Make RFDiffusion mAbs and VHHs with partial diffusion and see if numbering is recovered
- [ ] Make CIF and PDB both parseable
- [ ] Write tests for methods and setup CI/CD pipeline
- [ ] Add options to trim residues <1 and >128
- [ ] Make containers that can be pulled
- [ ] Make SoftAlign a git submodule; pick a specific commit, then overwrite PDB parsing logic to allow insertion codes
