# BIS Lab: Nature-Inspired Optimization Programs

This repository contains runnable Jupyter notebooks for six optimization
applications. Each notebook has an editable input section and saves a plot of
its result in the repository folder.

## Programs and outputs

| Application | Notebook | Plot output |
| --- | --- | --- |
| University timetable, genetic algorithm | `University_Timetable_GA.ipynb` | `university-timetable-ga.png` |
| University timetable, particle swarm optimization | `University_Timetable_PSO.ipynb` | `university-timetable-pso.png` |
| PCB routing, ant colony optimization | `PCB_Routing_ACO.ipynb` | `pcb-routing-aco.png` |
| Intrusion feature selection, cuckoo search | `NIDS_Feature_Selection_Cuckoo_Search.ipynb` | `nids-feature-selection-cuckoo-search.png` |
| Intrusion feature selection, grey wolf optimization | `NIDS_Feature_Selection_GWO.ipynb` | `nids-feature-selection-grey-wolf.png` |
| Solar panel placement, parallel cellular algorithm | `Solar_Panel_Placement_PCA.ipynb` | `solar-panel-placement-pca.png` |

Open a notebook in Jupyter and run its code cell. Edit the input values at the
top of the cell to change the problem instance, algorithm settings, or random
seed. The result is printed, displayed, and saved to the listed PNG file.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

The notebooks use NumPy and Matplotlib. The two intrusion detection examples
generate a deterministic synthetic binary dataset, so they run without a data
download. Their accuracy is a demonstration metric, not a benchmark on a real
intrusion detection corpus. Solar irradiance is also synthetic; the placement
notebook demonstrates the algorithm rather than modeling a specific site.

The original timetable PSO script, notebook examples, and unit tests are kept
alongside the six application notebooks.

## Original timetable PSO command-line program

The standalone PSO implementation remains available alongside the notebooks:

```bash
python3 university_timetable_pso.py --particles 60 --iterations 250 --seed 42 --output output.png
python3 -m unittest -v
```

It schedules six course sessions across five timeslots. The objective prioritizes
faculty clashes, room clashes, and room type errors, then reduces faculty gaps
and daily load imbalance. A zero score is a feasible global optimum for this
fixed example; other seeds or parameter settings may finish with a nonzero score.
