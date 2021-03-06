#!/bin/sh

python agent.py &
python agent.py --ao_sampling 0.25 &
python agent.py --ao_sampling 0.5 &
python agent.py --ao_sampling 1 &

python agent.py --env PaperMaze-v0 &
python agent.py --env PaperMaze-v0 --ao_sampling 0.25 &
python agent.py --env PaperMaze-v0 --ao_sampling 0.5 &
python agent.py --env PaperMaze-v0 --ao_sampling 1 &

python agent.py --env PaperMaze-v0 --options 12 &
python agent.py --env PaperMaze-v0 --ao_sampling 0.25 --options 12 &
python agent.py --env PaperMaze-v0 --ao_sampling 0.5 --options 12 &
python agent.py --env PaperMaze-v0 --ao_sampling 1 --options 12 &

python agent.py --env PaperMaze-v0 --iterations 5 &
python agent.py --env PaperMaze-v0 --ao_sampling 0.25 --iterations 5 &
python agent.py --env PaperMaze-v0 --ao_sampling 0.5 --iterations 5 &
python agent.py --env PaperMaze-v0 --ao_sampling 1 --iterations 5 &

python agent.py --env Maze-v0 --iterations 5 &
python agent.py --env Maze-v0 --ao_sampling 0.25 --iterations 5 &
python agent.py --env Maze-v0 --ao_sampling 0.5 --iterations 5 &
python agent.py --env Maze-v0 --ao_sampling 1 --iterations 5 &

python agent.py --env Maze-v0 --iterations 5 --options 16 &
python agent.py --env Maze-v0 --ao_sampling 0.25 --iterations 5 --options 16 &
python agent.py --env Maze-v0 --ao_sampling 0.5 --iterations 5 --options 16 &
python agent.py --env Maze-v0 --ao_sampling 1 --iterations 5 --options 16 &

wait
echo all experiments completed