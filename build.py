import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Simulation functions ---
def simulate_sir(N, beta, gamma, days):
    S, I, R, incidence = [N-1], [1], [0], [0]
    for _ in range(days):
        new_infections = beta * S[-1] * I[-1] / N
        new_recoveries = gamma * I[-1]

        S.append(S[-1] - new_infections)
        I.append(I[-1] + new_infections - new_recoveries)
        R.append(R[-1] + new_recoveries)
        incidence.append(incidence[-1] + new_infections)

    return np.array(S), np.array(I), np.array(R), np.diff(incidence, prepend=0)

def simulate_seir(N, beta, gamma, sigma, days):
    S, E, I, R, incidence = [N-1], [0], [1], [0], [0]
    for _ in range(days):
        new_exposed = beta * S[-1] * I[-1] / N
        new_infectious = sigma * E[-1]
        new_recoveries = gamma * I[-1]

        S.append(S[-1] - new_exposed)
        E.append(E[-1] + new_exposed - new_infectious)
        I.append(I[-1] + new_infectious - new_recoveries)
        R.append(R[-1] + new_recoveries)
        incidence.append(incidence[-1] + new_infectious)

    return np.array(S), np.array(I), np.array(R), np.diff(incidence, prepend=0)

def simulate_se2ir(N, beta, gamma, sigma, days):
    S, E1, E2, I, R, incidence = [N-1], [0], [0], [1], [0], [0]
    for _ in range(days):
        new_exposed = beta * S[-1] * I[-1] / N
        trans_E1_E2 = sigma * E1[-1]
        trans_E2_I = sigma * E2[-1]
        new_recoveries = gamma * I[-1]

        S.append(S[-1] - new_exposed)
        E1.append(E1[-1] + new_exposed - trans_E1_E2)
        E2.append(E2[-1] + trans_E1_E2 - trans_E2_I)
        I.append(I[-1] + trans_E2_I - new_recoveries)
        R.append(R[-1] + new_recoveries)
        incidence.append(incidence[-1] + trans_E2_I)

    return np.array(S), np.array(I), np.array(R), np.diff(incidence, prepend=0)

# --- Streamlit App ---
st.title("Epidemic Model Simulator")

models_selected = st.multiselect("Select model(s) to overlay:", ["SIR", "SEIR", "SE2IR"], default=["SIR"])

N = st.slider("Population size (N)", 100, 10000, 1000)
beta = st.slider("Transmission rate (beta)", 0.0, 5.0, 1.5, step=0.1)
gamma = st.slider("Recovery rate (gamma)", 0.1, 2.0, 0.5, step=0.1)
days = st.slider("Simulation days", 10, 200, 60)
sigma = st.slider("Incubation rate (sigma, for SEIR/SE2IR)", 0.1, 2.0, 0.5, step=0.1)

results = {}
for model in models_selected:
    if model == "SIR":
        S, I, R, inc = simulate_sir(N, beta, gamma, days)
        R0 = beta / gamma
    elif model == "SEIR":
        S, I, R, inc = simulate_seir(N, beta, gamma, sigma, days)
        R0 = beta / gamma
    elif model == "SE2IR":
        S, I, R, inc = simulate_se2ir(N, beta, gamma, sigma, days)
        R0 = beta / gamma
    results[model] = (S, I, R, inc, R0)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for model, (S, I, R, inc, R0) in results.items():
    axs[0].plot(S, label=f"{model} (R₀ ≈ {R0:.2f})")
    axs[1].plot(I, label=f"{model} (R₀ ≈ {R0:.2f})")
    axs[2].plot(R, label=f"{model} (R₀ ≈ {R0:.2f})")
    axs[3].plot(inc, label=f"{model} (R₀ ≈ {R0:.2f})")

axs[0].set_title("Susceptible over Time")
axs[1].set_title("Infected over Time")
axs[2].set_title("Recovered over Time")
axs[3].set_title("Incident Infections over Time")

for ax in axs:
    ax.set_xlabel("Days")
    ax.set_ylabel("Count")
    ax.legend()

plt.tight_layout()
st.pyplot(fig)
