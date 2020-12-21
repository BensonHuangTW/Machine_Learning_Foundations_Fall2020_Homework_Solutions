# Machine Learning Foundation Assignments Solutions
This repository contains solutions to the assignments of the course Machine Learning Foundations (HTML).
## Course Information
* Course Name: National Taiwan University CSIE5432 Machine Learning Foundations, Fall 2020
* Course Description: This course introduces the basics of learning theories, the design and analysis of learning algorithms, and some applications of machine learning.
* Instructor: Prof. [Hsuan-Tien Lin](https://www.csie.ntu.edu.tw/~htlin/)

For lectures and videos of this course, please refer to the [course website](https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/).
## Environments
The codes can be excuted in `python--3.6.12`. Basically, the codes in hw1 ~ hw3 just import `numpy` only and some figures in the solutions pdf files are generated using `matplotlib`. For hw4, you need to install `scikit-learn` and `liblinear` additionally. For details, please refer to [requirements.txt](./requirements.txt). You can use conda to install them: just run the following command in the repository directory,
```bash
$ conda install --yes --file requirements.txt
```
, or you can use pip:
```bash
$ pip install -r requirements.txt
```
## Executions
For example, if you want to see the results of exercise 16. of homework 1, please go to `./homework1/codes` and run the following command:
```bash
$ python3 16.py
```
## Solutions Links
* [homework1](./homework1/homework1_solution.pdf)
* [homework2](./homework2/homework2_solution.pdf)
* [homework3](./homework3/homework3_solution.pdf)
* [homework4](./homework4/homework4_solution.pdf)
