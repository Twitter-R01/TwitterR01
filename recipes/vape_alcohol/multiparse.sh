#!/bin/bash


# Parse 2019 data in parallel: 3-month batches
sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_alcohol/template_2019.par 20190101 20190399 &
sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_alcohol/template_2019.par 20190401 20190699 &
sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_alcohol/template_2019.par 20190701 20190999 &
sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_alcohol/template_2019.par 20191001 20191299 &

# Parse 2020 data in parallel: 3-month batches
sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_alcohol/template_2020.par 20200101 20200399 &
sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_alcohol/template_2020.par 20200401 20200699 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_alcohol/template_2020.par 20200701 20200999 &
