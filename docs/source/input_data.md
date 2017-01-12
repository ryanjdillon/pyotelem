# Input data

The code is currently implemented using current standards for Little Leonardo acceleromter data (meaning what is used by SMRU, University of Tokyo, and University of Tromsø).

All experiments to be used in an anlysis should be located in individual folders within one parent (or "root") data path. These experiment data folders should have the following naming pattern:

    `20160101_W190PD3GT_34839_Skinny_Control`
    `yyyymmdd_<tag model>_<tag id>_<animal id>_<experimental treatment>`

An example folder heirarchy:

```
my_project/
  all_experiment_data/
    20160101_W190PD3GT_34839_Skinny_Control/
      data_files.txt
      data_files.npy
      cal.yaml
      meta.yaml
    20160101_W190PD3GT_34839_Skinny_Control/
      ...
  glide_analysis/
    20160101_W190PD3GT_34839_Skinny_Control/
      figures/
      glides.npy
      glid_ratio.npy
      glide_analysis_config.yaml
    20160101_W190PD3GT_34839_Skinny_Control/
      ...
    experiments.yaml
    subglides.npy
    dives.npy
  mcmc_analysis/
    mcmc_config.yaml
    mcmc_results.yaml
```

