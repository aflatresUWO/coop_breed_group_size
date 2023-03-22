# coop_breed_group_size

Simulation code, written in Julia 1.7.1 and Python 3.8 to support the paper "Evolution of cooperative breeding with group size
effects and ecological feedbacks"

The files have been built and tested on Windows 10. If you are using macOS or Ubuntu, you may need to adapt some steps.

Software used: Jupyter notebook (anaconda3) and spyder (anaconda3).

To run data, open the julia file main.ipynb and run the different cells until "Simulations" section. This will compile the parameters and useful function to create the data. You can now run the simulations cells to create csv files compiling the data. Four data files are created. Those four files are used in the python file main.py to create the different figures of our article.

To create the figures, run the python file main.py after creating the csv files. You can modify the parameters in main.ipynb file to create new plots.
