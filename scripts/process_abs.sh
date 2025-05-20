cd /home/delalamo/sabdab_structures/parsed/aho/human_L && \
    /home/delalamo/rosetta/source/bin/antibody_numbering_converter.linuxgccrelease \
    -in:file:s /home/delalamo/sabdab_structures/parsed/fixed_chains/human_L/*pdb \
    -input_ab_scheme IMGT -output_ab_scheme AHo
cd /home/delalamo/sabdab_structures/parsed/aho/human_H && \
    /home/delalamo//rosetta/source/bin/antibody_numbering_converter.linuxgccrelease \
    -in:file:s /home/delalamo/sabdab_structures/parsed/fixed_chains//human_H/*pdb \
    -input_ab_scheme IMGT -output_ab_scheme AHo
cd /home/delalamo/sabdab_structures/parsed/aho/mouse_H && \
    /home/delalamo//rosetta/source/bin/antibody_numbering_converter.linuxgccrelease \
    -in:file:s /home/delalamo/sabdab_structures/parsed/fixed_chains//mouse_H/*pdb \
    -input_ab_scheme IMGT -output_ab_scheme AHo
cd /home/delalamo/sabdab_structures/parsed/aho/mouse_L && \
    /home/delalamo//rosetta/source/bin/antibody_numbering_converter.linuxgccrelease \
    -in:file:s /home/delalamo/sabdab_structures/parsed/fixed_chains/mouse_L/*pdb \
    -input_ab_scheme IMGT -output_ab_scheme AHo