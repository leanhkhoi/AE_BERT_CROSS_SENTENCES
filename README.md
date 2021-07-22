# Adversarial Training for Aspect-Based Sentiment Analysis with BERT
Code for "[Adversarial Training for Aspect-Based Sentiment Analysis with BERT](https://arxiv.org/pdf/2001.11316)".

We have used the codebase from the following paper and improved upon their results by applying adversarial training.
"[BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://www.aclweb.org/anthology/N19-1242.pdf)".


## Running

Place laptop and restaurant post-trained BERTs into ```pt_model/laptop_pt``` and ```pt_model/rest_pt```, respectively. The post-trained Laptop weights can be download [here](https://drive.google.com/file/d/1io-_zVW3sE6AbKgHZND4Snwh-wi32L4K/view?usp=sharing) and restaurant [here](https://drive.google.com/file/d/1TYk7zOoVEO8Isa6iP0cNtdDFAUlpnTyz/view?usp=sharing).

Execute the following command to run the model for Aspect Extraction task:

```script\run_ae.bat ae laptop_pt laptop pt_ae 9```

Here, ```laptop_pt``` is the post-trained weights for laptop, ```laptop``` is the domain, ```pt_ae``` is the fine-tuned folder in ```run/```, ```9``` means run 9 times.

Similarly,
```
script\run_ae.bat ae rest_pt rest pt_ae 9
```
### Evaluation

Execute the following command to evaluate the model for Aspect Extraction task:

```eval\run_ae_eval.bat laptop 9```

Here ```laptop``` is the domain, ```9``` means run 9 predictions corresponding to 9 runs

The evaluation additionally needs Java JRE/JDK to be installed.

Open ```result.ipynb``` and check the results.

## Citation

```
@misc{karimi2020adversarial,
    title={Adversarial Training for Aspect-Based Sentiment Analysis with BERT},
    author={Akbar Karimi and Leonardo Rossi and Andrea Prati and Katharina Full},
    year={2020},
    eprint={2001.11316},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
