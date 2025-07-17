"##############################################################################"
"######################## Importing important packages ########################"
"##############################################################################"

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.cm as cm
from scipy.optimize import minimize

np.random.seed(1998)


"##############################################################################"
"###################### Model-Based Design of Experiments #####################"
"##############################################################################"

def SR_model(z0, equations, t, t_eval):
    i = 0

    for equation in equations:
        equation = str(equation)
        equation = equation.replace("CNO", "z[0]")
        equation = equation.replace("N", "z[1]")
        equation = equation.replace("H", "z[2]")
        equations[i] = equation
        i += 1

    def nest(t, z):
        dNOdt = (1) * eval(str(equations[0]))
        dNdt = (-1) * eval(str(equations[0]))
        dHdt = (-1/2) * eval(str(equations[0]))
        dzdt = [dNOdt, dNdt, dHdt]
        return dzdt

    sol = solve_ivp(nest, t, z0, t_eval = t_eval, method = "RK45")  

    return sol.y

def MBDoE(ic, time, sym_model_1, sym_model_2):
    timesteps = len(time)
    SR_thing_1 = SR_model(ic, sym_model_1, [0, np.max(time)], list(time))
    SR_thing_1 = SR_thing_1.reshape(len(time), -1)
    SR_thing_2 = SR_model(ic, sym_model_2, [0, np.max(time)], list(time))
    SR_thing_2 = SR_thing_2.reshape(len(time), -1)
    difference = -np.sum((SR_thing_1 - SR_thing_2)**2)
    return difference

def Opt_Rout(multistart, number_parameters, lower_bound, upper_bound, to_opt, \
    time, sym_model_1, sym_model_2):
    localsol = np.empty([multistart, number_parameters])
    localval = np.empty([multistart, 1])
    boundss = tuple([(lower_bound[i], upper_bound[i]) for i in range(len(lower_bound))])
    
    for i in range(multistart):
        x0 = np.random.uniform(lower_bound, upper_bound, size = number_parameters)
        res = minimize(to_opt, x0, args = (time, sym_model_1, sym_model_2), \
                        method = 'L-BFGS-B', bounds = boundss)
        localsol[i] = res.x
        localval[i] = res.fun

    minindex = np.argmin(localval)
    opt_val = localval[minindex]
    opt_param = localsol[minindex]
    
    return opt_val, opt_param


"##############################################################################"
"########################## MBDoE on Competing Models #########################"
"##############################################################################"

multistart = 1
number_parameters = 3
lower_bound = np.array([0 , 0, 0])
upper_bound = np.array([10, 2, 3])
to_opt = MBDoE
timesteps = 15
time = np.linspace(0, 10, timesteps)

sym_model_1 = list((
    '0.013822312359624923 - 0.3736778093978375*CNO',
))

sym_model_2 = list((
    '-0.37019046699209807*CNO',
))

real_model = list((
    '(-2*CNO**2)/(1+5*CNO)',
))

a, b = Opt_Rout(multistart, number_parameters, lower_bound, upper_bound, to_opt, \
    time, sym_model_1, sym_model_2)

print('Optimal experiment: ', b)


"##############################################################################"
"########################### Plot MBDoE Experiment ############################"
"##############################################################################"

Title = "MBDoE Second Best SR Model vs Real Model"
species = ["NO", "N", "H"]

STD = 0.
noise = np.random.normal(0, STD, size = (number_parameters, timesteps))

y   = SR_model(b, sym_model_1 , [0, np.max(time)], list(time))
yyy = SR_model(b, sym_model_2, [0, np.max(time)], list(time))
truth = SR_model(b, real_model, [0, np.max(time)], list(time))

fig, ax = plt.subplots()
ax.set_title(Title)
ax.set_ylabel("Concentration $(M)$")
ax.set_xlabel("Time $(h)$")

color_1 = cm.viridis(np.linspace(0, 1, number_parameters))
color_2 = cm.Wistia(np.linspace(0, 1, number_parameters))
color_3 = cm.cool(np.linspace(0, 1, number_parameters))

for j in range(number_parameters):
    ax.plot(time, truth[j], "x", markersize = 3, color = color_1[j])
    ax.plot(time, y[j], color = color_1[j], label = str('SR Model 1 - ' + str(species[j])))
    ax.plot(time, yyy[j], linestyle = 'dashed', color = color_1[j], label = str('SR Model 2 - ' + str(species[j])))

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.grid(alpha = 0.5)
ax.legend()
plt.show()

print(np.sum((y - yyy)**2))