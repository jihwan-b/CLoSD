# Part of Robogo(YAICON project)


run commands at two seperate terminal windows


### 1. Run Closd text to motion
```
cd CLoSD
```
```
python closd/run.py  learning=im_big robot=smpl_humanoid  epoch=-1 test=True no_virtual_display=True  headless=False env.num_envs=1  env=closd_t2m exp_name=CLoSD_t2m_finetune  env.episode_length=400
```

### 2. Run prompt injector
```
python closd/custom_t2m/prompt_injector.py
```


# Original Repository:
https://github.com/GuyTevet/CLoSD

