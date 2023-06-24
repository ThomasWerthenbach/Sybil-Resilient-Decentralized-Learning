# Towards Sybil Resilience in Decentralized Learning

This repository contains the code for the paper "Towards Sybil Resilience in Decentralized Learning" by Thomas Werthenbach and Johan Pouwelse.

### Usage

1. Ensure that you have installed the dependencies listed in `requirements.txt`.
2. Ensure that you have installed all Gumby dependencies as described in the [Gumby repository](https://github.com/Tribler/gumby).
3. Ensure that you have forked the [Py-IPv8 repository](https://github.com/Tribler/py-ipv8).
4. Run the experiment using:<br>```IPV8_DIR=/path/to/ipv8/fork gumby/run.py gumby/experiments/DL_NIID/exp.conf``` 

The experiment configurations can be found in the `gumby/experiments` directory. The `settings.json` files contains the configuration for the experiment. The `das_exp.conf` files contain the configuration for the DAS-6 supercomputer.  
