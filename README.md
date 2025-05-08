# SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation

Updating...

<p align="center">
  <!-- <img src="assets/teaser.png" alt="Overview of Batch3DMOT architecture" width="245" /> -->
  <img src="assets/image.png" alt="Overview of ThinkPlan" width="1200" />
</p>

## Code overview

SmallPlan adopts the open-sourced code from MoMaLLM [[**paper**](https://arxiv.org/abs/2403.08605)] and TextGames [[**paper**](https://arxiv.org/abs/2502.18431)]

* We use MoMaLLM code base for simulation and LLM agent set-up. We then add functions for fine-tuning the SLMs with SFT and RL. The training and reward functions are provided in ```./moma_llm/env/llm_env.py```

* We use TextGames for Out-of-domain Analysis experiment. In their code base, we mostly run ```./textgames/agents/dsr1_distill.py``` to test our fine-tuned SLMs. 

The weight of the fine-tunned SLMs can be found [**here**](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/cuong_pham_mbzuai_ac_ae/EuZoi_N-OvtEsRHnlMObw0UB0WmpykeMQTjKOxMcMKbFjw?e=1blu1h)

## Docker
For simple use, we provide a Dockerfile and Vscode devcontainer configuration. 

This requires [Docker](https://www.docker.com/) and [Vscode](https://code.visualstudio.com/) to be installed, as well as the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU support.
Within vscode, please install the Docker extension. Alternatively, use the same arguments as defined in `devcointainer.json` to build and run the container without vscode.

1. Please add `OPENAI_ORGANIZATION` and `OPENAI_API_KEY` to `devcontainer.json` to use the OpenAI API. Doing so will incur costs! Alternatively, set the `agent` value in `configs/moma_llm.yaml` to `random` or `greedy` to use a baseline that does not use the API.
2. Download data locally and store in [repository root]/data/ (will be mounted into container): https://stanfordvl.github.io/iGibson/dataset.html
3. Open project in vscode. Install the vscode docker extension. Then press `F1` -> `Build and reopen in container` to open the project in docker.
4. Test igibson. If not activated, first activate the conda environment with `source activate igibson`. Then, this should open a window with the camera, robot moving around: `python -m igibson.examples.environments.env_nonint_example`. Note that this requires a display. Make sure that the `DISPLAY` environment variable is defined (usually same value as outside the docker container).
5. On your host system, you might have to run: `xhost +local:root` or `sudo xhost +` otherwise you might observe an error like `qt.qpa.xcb: could not connect to display :0`
6. Download the assets locally (within docker to have the igibson dependencies): run `python -m igibson.utils.assets_utils --download_assets` - this will persist in the mounted directory

This docker image builds up on the iGibson docker image. For more information on how to use iGibson with docker, see: https://stanfordvl.github.io/iGibson/quickstart.html?highlight=docker.


## Local installation
To use without docker, instead 
1. Install iGibson: https://stanfordvl.github.io/iGibson/installation.html
2. Update the conda environment for igibson with the dependencies with `pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113`
3. Follow the data-download steps from the Docker section above

<!-- ## Running the approach
Running the GPT_4 using python3 main.py 

Running the Local LLM: python3 test.py 

Running the training: python3 train.py -->

## Citation

Please cite our work if you find helpful.

```
@misc{pham2025smallplanleveragesmalllanguage,
      title={SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation}, 
      author={Quang P. M. Pham and Khoi T. N. Nguyen and Nhi H. Doan and Cuong A. Pham and Kentaro Inui and Dezhen Song},
      year={2025},
      eprint={2505.00831},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.00831}, 
}
```