#!/bin/bash

# Run two months at a time (1 node each) to balance processing requirements
# Each year of data is stored in a separate directory so each year uses different parser template:

# REMEMBER TO NOT PARSE THE SAME DATES TWICE WITHOUT REMOVING OLD OUTPUT (output files append, not overwrite)


#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2017.par 20170100 20170299 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2017.par 20170300 20170499 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2017.par 20170500 20170699 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2017.par 20170700 20170899 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2017.par 20170900 20171099 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2017.par 20171100 20171299 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2018.par 20180100 20180299 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2018.par 20180300 20180499 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2018.par 20180500 20180699 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2018.par 20180700 20180899 &

# This is around when the classifier was trained (2018-08-17 through 2018-10-19)
sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2018.par 20180900 20181099 &

#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2018.par 20181100 20181299 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2019.par 20190100 20190299 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2019.par 20190300 20190499 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2019.par 20190500 20190699 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2019.par 20190700 20190899 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2019.par 20190900 20191099 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2019.par 20191100 20191299 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2020.par 20200100 20200299 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2020.par 20200300 20200499 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2020.par 20200500 20200699 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2020.par 20200700 20200899 &
#sbatch multiparser.sb /pylon5/be5fpap/jcolditz/UArk/recipes/vape_commercial/vape_2020.par 20200900 20201099 &
