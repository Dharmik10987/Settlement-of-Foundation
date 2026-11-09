
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Foundation Settlement Calculator",
    page_icon="🏗️",
    layout="wide"
)

st.title("FOUNDATION SETTLEMENT CALCULATOR")
st.subheader(
    "Immediate, Primary Consolidation and Secondary Settlement"
)



# =========================================================
# 1. SOIL PRESETS
# =========================================================

SOILS = {
    "Loose sand": {
        "gamma": 17.0, "Es": 10000.0, "nu": 0.30,
        "Cc": 0.0, "Cr": 0.0, "C_alpha": 0.0,
        "e0": 0.65, "ep": 0.65, "sigma_pc": 100.0
    },
    "Medium-dense sand": {
        "gamma": 18.0, "Es": 25000.0, "nu": 0.30,
        "Cc": 0.0, "Cr": 0.0, "C_alpha": 0.0,
        "e0": 0.60, "ep": 0.60, "sigma_pc": 100.0
    },
    "Dense sand": {
        "gamma": 19.0, "Es": 50000.0, "nu": 0.28,
        "Cc": 0.0, "Cr": 0.0, "C_alpha": 0.0,
        "e0": 0.50, "ep": 0.50, "sigma_pc": 100.0
    },
    "Baseline sand with fines": {
        "gamma": 18.0, "Es": 20000.0, "nu": 0.30,
        "Cc": 0.15, "Cr": 0.03, "C_alpha": 0.003,
        "e0": 0.65, "ep": 0.60, "sigma_pc": 100.0
    },
    "Soft clay": {
        "gamma": 16.5, "Es": 4000.0, "nu": 0.45,
        "Cc": 0.35, "Cr": 0.07, "C_alpha": 0.025,
        "e0": 1.00, "ep": 0.90, "sigma_pc": 50.0
    },
    "Stiff clay": {
        "gamma": 19.0, "Es": 30000.0, "nu": 0.35,
        "Cc": 0.20, "Cr": 0.04, "C_alpha": 0.010,
        "e0": 0.65, "ep": 0.60, "sigma_pc": 200.0
    },
    "Silty soil": {
        "gamma": 18.0, "Es": 12000.0, "nu": 0.32,
        "Cc": 0.12, "Cr": 0.025, "C_alpha": 0.006,
        "e0": 0.70, "ep": 0.65, "sigma_pc": 100.0
    },
    "Layered soil": {
        "gamma": 17.0, "Es": 10000.0, "nu": 0.30,
        "Cc": 0.35, "Cr": 0.07, "C_alpha": 0.025,
        "e0": 1.00, "ep": 0.90, "sigma_pc": 50.0
    }
}

# =========================================================
# 2. CALCULATION FUNCTIONS
# =========================================================

def calculate_net_pressure(P, B, L, gamma, Df):
    area = B * L
    q_gross = P / area
    q_surcharge = gamma * Df
    q_net = max(0.0, q_gross - q_surcharge)
    return q_gross, q_net


def calculate_immediate(q_net, B, Es, nu, Is):
    """Simplified elastic immediate settlement. Output in metres."""
    return q_net * B * (1 - nu**2) * Is / Es


def calculate_stress_increase(q_net, B, L, z):
    """Approximate 2:1 stress distribution. Output in kPa."""
    return q_net * B * L / ((B + z) * (L + z))


def calculate_compression_index(e1, e2, sigma1, sigma2):
    """
    Calculate compression index from two e-log effective stress readings.
    Returns None for invalid inputs.
    """
    if sigma1 <= 0 or sigma2 <= 0 or sigma2 <= sigma1:
        return None

    if e1 < 0 or e2 < 0 or e1 < e2:
        return None

    denominator = np.log10(sigma2 / sigma1)

    if denominator <= 0:
        return None

    return (e1 - e2) / denominator


def calculate_primary(
    H, sigma0, delta_sigma, sigma_pc, Cc, Cr, e0
):
    """
    One-dimensional e-log effective stress calculation.

    NC: virgin compression index Cc.
    OC: recompression index Cr below sigma_pc,
        compression index Cc above sigma_pc.

    Output in metres.
    """

    if H <= 0 or sigma0 <= 0 or e0 <= -1:
        return 0.0

    final_stress = sigma0 + delta_sigma

    if final_stress <= sigma0:
        return 0.0

    if sigma_pc <= sigma0:
        settlement = (
            Cc * H / (1 + e0)
            * np.log10(final_stress / sigma0)
        )

    elif final_stress <= sigma_pc:
        settlement = (
            Cr * H / (1 + e0)
            * np.log10(final_stress / sigma0)
        )

    else:
        S_recompression = (
            Cr * H / (1 + e0)
            * np.log10(sigma_pc / sigma0)
        )

        e_pc = (
            e0 - Cr * np.log10(sigma_pc / sigma0)
        )

        if e_pc <= -1:
            raise ValueError(
                "Calculated void ratio at preconsolidation "
                "stress is invalid. Check input parameters."
            )

        S_virgin = (
            Cc * H / (1 + e_pc)
            * np.log10(final_stress / sigma_pc)
        )

        settlement = S_recompression + S_virgin

    return max(0.0, settlement)


def calculate_secondary(H, C_alpha, ep, t1, t2):
    """Secondary compression settlement. Output in metres."""

    if H <= 0 or C_alpha <= 0 or ep <= -1:
        return 0.0

    if t1 <= 0 or t2 <= t1:
        return 0.0

    return (
        C_alpha * H / (1 + ep)
        * np.log10(t2 / t1)
    )


def calculate_layer_settlement(
    layers, q_net, B, L, t1, t2,
    use_primary=True, use_secondary=True
):
    """Sum primary and secondary settlement across layers."""

    primary_total = 0.0
    secondary_total = 0.0
    layer_results = []

    for _, layer in layers.iterrows():

        H = float(layer["Thickness H (m)"])
        z = float(layer["Centre depth z (m)"])

        sigma0 = float(
            layer["Initial effective stress (kPa)"]
        )

        sigma_pc = float(
            layer["Preconsolidation stress (kPa)"]
        )

        Cc = float(layer["Cc"])
        Cr = float(layer["Cr"])
        C_alpha = float(layer["C_alpha"])
        e0 = float(layer["Initial void ratio e0"])
        ep = float(layer["End-primary void ratio ep"])

        delta_sigma = calculate_stress_increase(
            q_net, B, L, z
        )

        Sc = 0.0
        Ss = 0.0

        if use_primary and bool(layer["Include primary"]):
            Sc = calculate_primary(
                H, sigma0, delta_sigma,
                sigma_pc, Cc, Cr, e0
            )

        if use_secondary and bool(layer["Include secondary"]):
            Ss = calculate_secondary(
                H, C_alpha, ep, t1, t2
            )

        primary_total += Sc
        secondary_total += Ss

        layer_results.append({
            "Layer": layer["Layer"],
            "Stress increase (kPa)": delta_sigma,
            "Cc": Cc,
            "Cr": Cr,
            "Primary settlement (mm)": Sc * 1000,
            "Secondary settlement (mm)": Ss * 1000
        })

    return primary_total, secondary_total, layer_results


# =========================================================
# 3. FOUNDATION INPUTS
# =========================================================

st.header("SECTION A - FOUNDATION DETAILS")

c1, c2, c3 = st.columns(3)

with c1:
    P = st.number_input(
        "Service vertical load P (kN)",
        min_value=1.0,
        value=900.0
    )

    Df = st.number_input(
        "Foundation depth Df (m)",
        min_value=0.0,
        value=1.5
    )

with c2:
    L = st.number_input(
        "Footing length L (m)",
        min_value=0.1,
        value=2.0
    )

    settlement_limit = st.number_input(
        "Allowable total settlement (mm)",
        min_value=0.1,
        value=25.0
    )

with c3:
    B_start = st.number_input(
        "Minimum breadth (m)",
        min_value=0.1,
        value=1.5
    )

    B_end = st.number_input(
        "Maximum breadth (m)",
        min_value=0.2,
        value=4.0
    )

    B_step = st.number_input(
        "Breadth increment (m)",
        min_value=0.05,
        value=0.5
    )


# =========================================================
# 4. SOIL INPUTS
# =========================================================

st.header("SECTION B - SOIL PARAMETERS")

soil_name = st.selectbox(
    "Select soil scenario",
    list(SOILS.keys())
)

preset = SOILS[soil_name]

c1, c2, c3 = st.columns(3)

with c1:
    gamma = st.number_input(
        "Bulk unit weight γ (kN/m³)",
        min_value=1.0,
        value=float(preset["gamma"])
    )

    Es = st.number_input(
        "Elastic modulus Es (kPa)",
        min_value=1.0,
        value=float(preset["Es"])
    )

with c2:
    nu = st.number_input(
        "Poisson's ratio ν",
        min_value=0.0,
        max_value=0.49,
        value=float(preset["nu"])
    )

    Is = st.number_input(
        "Immediate settlement influence factor Is",
        min_value=0.01,
        value=1.12
    )

with c3:
    water_table = st.number_input(
        "Water table depth below ground (m)",
        min_value=0.0,
        value=3.0
    )

    gamma_w = st.number_input(
        "Water unit weight (kN/m³)",
        min_value=1.0,
        value=9.81
    )

st.caption(
    "The water-table depth is recorded but does not automatically "
    "calculate effective stresses. Enter groundwater-consistent "
    "initial effective stresses for each consolidation layer."
)


# =========================================================
# 5. SOIL LAYER INPUTS
# =========================================================

st.header("SECTION C - COMPRESSIBLE SOIL LAYERS")

st.write(
    "Depth to layer centre is measured below the footing base. "
    "Enter each layer's properties separately."
)

if soil_name == "Layered soil":

    default_layers = pd.DataFrame([
        {
            "Layer": "Upper loose sand",
            "Thickness H (m)": 0.5,
            "Centre depth z (m)": 0.25,
            "Initial effective stress (kPa)": 25.0,
            "Preconsolidation stress (kPa)": 100.0,
            "C_alpha": 0.0,
            "Initial void ratio e0": 0.65,
            "End-primary void ratio ep": 0.65,
            "Include primary": False,
            "Include secondary": False
        },
        {
            "Layer": "Lower soft clay",
            "Thickness H (m)": 3.0,
            "Centre depth z (m)": 2.0,
            "Initial effective stress (kPa)": 45.0,
            "Preconsolidation stress (kPa)": 60.0,
            "C_alpha": 0.025,
            "Initial void ratio e0": 1.0,
            "End-primary void ratio ep": 0.90,
            "Include primary": True,
            "Include secondary": True
        }
    ])

else:

    default_layers = pd.DataFrame([
        {
            "Layer": "Layer 1",
            "Thickness H (m)": 3.0,
            "Centre depth z (m)": 1.5,
            "Initial effective stress (kPa)": 50.0,
            "Preconsolidation stress (kPa)": preset["sigma_pc"],
            "C_alpha": preset["C_alpha"],
            "Initial void ratio e0": preset["e0"],
            "End-primary void ratio ep": preset["ep"],
            "Include primary": preset["Cc"] > 0,
            "Include secondary": preset["C_alpha"] > 0
        }
    ])

edited_layers = st.data_editor(
    default_layers,
    num_rows="dynamic",
    use_container_width=True,
    key="layers_editor"
)

layers = edited_layers.copy()


# =========================================================
# 6. Cc AND Cr LABORATORY INPUTS
# =========================================================

st.subheader("Cc and Cr Calculation from Consolidation Test")


def make_test_defaults(layer_df, soil_preset):
    """Create illustrative editable test readings for each layer."""

    rows = []

    for _, layer in layer_df.iterrows():

        e0 = float(layer["Initial void ratio e0"])
        pc = float(layer["Preconsolidation stress (kPa)"])

        name = str(layer["Layer"])

        if "sand" in name.lower():
            cc_default = 0.0
            cr_default = 0.0
        else:
            cc_default = float(soil_preset["Cc"])
            cr_default = float(soil_preset["Cr"])

        # Virgin compression readings: both stresses above sigma_pc.
        cc_s1 = pc * 1.2
        cc_s2 = pc * 12.0

        cc_e1 = e0
        cc_e2 = e0 - cc_default * np.log10(cc_s2 / cc_s1)

        # Recompression readings: both stresses below sigma_pc.
        cr_s1 = pc * 0.1
        cr_s2 = pc * 0.8

        cr_e1 = e0
        cr_e2 = e0 - cr_default * np.log10(cr_s2 / cr_s1)

        rows.append({
            "Layer": name,
            "Cc e1": cc_e1,
            "Cc e2": cc_e2,
            "Cc σ1' (kPa)": cc_s1,
            "Cc σ2' (kPa)": cc_s2,
            "Cr e1": cr_e1,
            "Cr e2": cr_e2,
            "Cr σ1' (kPa)": cr_s1,
            "Cr σ2' (kPa)": cr_s2
        })

    return pd.DataFrame(rows)


test_defaults = make_test_defaults(layers, preset)

test_inputs = st.data_editor(
    test_defaults,
    num_rows="fixed",
    use_container_width=True,
    key="consolidation_test_editor",
    disabled=["Layer"]
)

# Calculate Cc and Cr for every layer.

calculated_cc = []
calculated_cr = []
test_errors = []

for index, row in test_inputs.iterrows():

    layer_name = row["Layer"]

    cc = calculate_compression_index(
        float(row["Cc e1"]),
        float(row["Cc e2"]),
        float(row["Cc σ1' (kPa)"]),
        float(row["Cc σ2' (kPa)"])
    )

    cr = calculate_compression_index(
        float(row["Cr e1"]),
        float(row["Cr e2"]),
        float(row["Cr σ1' (kPa)"]),
        float(row["Cr σ2' (kPa)"])
    )

    # Check whether the readings belong to the correct curve portions.
    pc = float(
        layers.iloc[index]["Preconsolidation stress (kPa)"]
    )

    cc_s1 = float(row["Cc σ1' (kPa)"])
    cc_s2 = float(row["Cc σ2' (kPa)"])

    cr_s1 = float(row["Cr σ1' (kPa)"])
    cr_s2 = float(row["Cr σ2' (kPa)"])

    if cc is None:
        test_errors.append(
            f"{layer_name}: Invalid Cc readings. Check e1, e2, "
            "and ensure σ2' is greater than σ1'."
        )

    elif not (cc_s1 > pc and cc_s2 > pc):
        test_errors.append(
            f"{layer_name}: Cc stresses should both be above "
            "preconsolidation stress."
        )

    if cr is None:
        test_errors.append(
            f"{layer_name}: Invalid Cr readings. Check e1, e2, "
            "and ensure σ2' is greater than σ1'."
        )

    elif not (cr_s1 < pc and cr_s2 < pc):
        test_errors.append(
            f"{layer_name}: Cr stresses should both be below "
            "preconsolidation stress."
        )

    calculated_cc.append(cc)
    calculated_cr.append(cr)


# Display calculated indices.

st.subheader("Calculated Compression Indices")

index_results = pd.DataFrame({
    "Layer": test_inputs["Layer"],
    "Calculated Cc": calculated_cc,
    "Calculated Cr": calculated_cr
})

st.dataframe(
    index_results,
    use_container_width=True
)

if test_errors:
    for error in test_errors:
        st.error(error)

# Add the calculated indices to the layer table.
# These values are used directly in settlement calculations.

layers["Cc"] = calculated_cc
layers["Cr"] = calculated_cr

st.success(
    "Cc and Cr are calculated from the entered laboratory readings "
    "and are automatically linked to the primary consolidation calculation."
)


# =========================================================
# 7. TIME INPUTS
# =========================================================

st.header("SECTION D - CONSOLIDATION TIME")

c1, c2 = st.columns(2)

with c1:
    t1 = st.number_input(
        "Time at end of primary consolidation t1",
        min_value=0.001,
        value=1.0
    )

with c2:
    t2 = st.number_input(
        "Final evaluation time t2",
        min_value=0.001,
        value=10.0
    )

st.caption(
    "Use the same time unit for both values, such as years."
)


# =========================================================
# 8. VALIDATION
# =========================================================

required_columns = [
    "Layer",
    "Thickness H (m)",
    "Centre depth z (m)",
    "Initial effective stress (kPa)",
    "Preconsolidation stress (kPa)",
    "Cc", "Cr", "C_alpha",
    "Initial void ratio e0",
    "End-primary void ratio ep",
    "Include primary",
    "Include secondary"
]

valid = True

if not all(col in layers.columns for col in required_columns):
    st.error("Layer table is missing required columns.")
    valid = False

elif layers.empty:
    st.error("Enter at least one compressible layer.")
    valid = False

else:

    numeric_columns = [
        "Thickness H (m)",
        "Centre depth z (m)",
        "Initial effective stress (kPa)",
        "Preconsolidation stress (kPa)",
        "Cc", "Cr", "C_alpha",
        "Initial void ratio e0",
        "End-primary void ratio ep"
    ]

    if layers[numeric_columns].isnull().any().any():
        st.error("Fill all numerical layer properties.")
        valid = False

    elif (layers["Thickness H (m)"] <= 0).any():
        st.error("Layer thickness must be positive.")
        valid = False

    elif (layers["Centre depth z (m)"] < 0).any():
        st.error("Layer centre depth cannot be negative.")
        valid = False

    elif (layers["Initial effective stress (kPa)"] <= 0).any():
        st.error("Initial effective stress must be positive.")
        valid = False

    elif (layers["Preconsolidation stress (kPa)"] <= 0).any():
        st.error("Preconsolidation stress must be positive.")
        valid = False

    elif (layers[["Cc", "Cr", "C_alpha"]] < 0).any().any():
        st.error("Compression indices cannot be negative.")
        valid = False

    elif (layers["Initial void ratio e0"] <= -1).any():
        st.error("Initial void ratio must be greater than -1.")
        valid = False

    elif (layers["End-primary void ratio ep"] <= -1).any():
        st.error("End-primary void ratio must be greater than -1.")
        valid = False

if test_errors:
    valid = False

if t2 <= t1:
    st.error("Final time t2 must be greater than t1.")
    valid = False

if B_end < B_start:
    st.error("Maximum breadth must exceed minimum breadth.")
    valid = False

if B_step <= 0:
    st.error("Breadth increment must be positive.")
    valid = False


# =========================================================
# 10. INDIVIDUAL CALCULATORS
# =========================================================

st.header("SECTION F - CALCULATE INDIVIDUAL SETTLEMENTS")

calc_B = st.number_input(
    "Breadth for individual calculations (m)",
    min_value=0.1,
    value=2.0,
    key="individual_B"
)

if valid:

    qgross_ind, qnet_ind = calculate_net_pressure(
        P, calc_B, L, gamma, Df
    )

    st.write(
        f"Gross pressure: **{qgross_ind:.2f} kPa**"
    )

    st.write(
        f"Net pressure: **{qnet_ind:.2f} kPa**"
    )

    with st.expander("Calculate Immediate Settlement", expanded=True):

        if st.button(
            "Calculate Immediate Settlement",
            key="btn_immediate"
        ):

            Si = calculate_immediate(
                qnet_ind, calc_B, Es, nu, Is
            )

            st.success(
                f"Immediate settlement = {Si * 1000:.2f} mm"
            )

    with st.expander("Calculate Primary Consolidation", expanded=True):

        if st.button(
            "Calculate Primary Settlement",
            key="btn_primary"
        ):

            Sc, _, layer_details = calculate_layer_settlement(
                layers, qnet_ind, calc_B, L,
                t1, t2,
                use_primary=True,
                use_secondary=False
            )

            st.success(
                f"Primary consolidation settlement = {Sc * 1000:.2f} mm"
            )

            st.dataframe(
                pd.DataFrame(layer_details),
                use_container_width=True
            )

    with st.expander("Calculate Secondary Consolidation", expanded=True):

        if st.button(
            "Calculate Secondary Settlement",
            key="btn_secondary"
        ):

            _, Ss, layer_details = calculate_layer_settlement(
                layers, qnet_ind, calc_B, L,
                t1, t2,
                use_primary=False,
                use_secondary=True
            )

            st.success(
                f"Secondary settlement = {Ss * 1000:.2f} mm"
            )

            st.dataframe(
                pd.DataFrame(layer_details),
                use_container_width=True
            )

else:
    st.warning(
        "Correct the input errors before performing settlement calculations."
    )


# =========================================================
# 11. BREADTH-WISE SETTLEMENT ANALYSIS
# =========================================================

st.header("SECTION G - SETTLEMENT FOR DIFFERENT FOOTING BREADTHS")

st.write(
    "This analysis calculates all settlement components for each "
    "trial breadth using the same selected footing length."
)

if st.button("Run Complete Breadth Analysis"):

    if not valid:

        st.error("Correct the input errors before running analysis.")

    else:

        breadths = np.arange(
            B_start,
            B_end + B_step * 0.5,
            B_step
        )

        results = []

        for B in breadths:

            qgross, qnet = calculate_net_pressure(
                P, B, L, gamma, Df
            )

            Si = calculate_immediate(
                qnet, B, Es, nu, Is
            )

            Sc, Ss, layer_details = calculate_layer_settlement(
                layers, qnet, B, L,
                t1, t2,
                use_primary=True,
                use_secondary=True
            )

            Si_mm = Si * 1000
            Sc_mm = Sc * 1000
            Ss_mm = Ss * 1000

            total = Si_mm + Sc_mm + Ss_mm

            status = (
                "PASS"
                if total <= settlement_limit
                else "FAIL"
            )

            results.append({
                "Breadth (m)": B,
                "Length (m)": L,
                "Area (m²)": B * L,
                "Gross pressure (kPa)": qgross,
                "Net pressure (kPa)": qnet,
                "Immediate (mm)": Si_mm,
                "Primary (mm)": Sc_mm,
                "Secondary (mm)": Ss_mm,
                "Total settlement (mm)": total,
                "Allowable (mm)": settlement_limit,
                "Status": status
            })

        df = pd.DataFrame(results)

        st.session_state["settlement_results"] = df


# =========================================================
# 12. DISPLAY RESULTS
# =========================================================

if "settlement_results" in st.session_state:

    df = st.session_state["settlement_results"]

    st.subheader("Breadth-wise Settlement Results")

    st.dataframe(
        df.style.format({
            "Breadth (m)": "{:.2f}",
            "Length (m)": "{:.2f}",
            "Area (m²)": "{:.2f}",
            "Gross pressure (kPa)": "{:.2f}",
            "Net pressure (kPa)": "{:.2f}",
            "Immediate (mm)": "{:.2f}",
            "Primary (mm)": "{:.2f}",
            "Secondary (mm)": "{:.2f}",
            "Total settlement (mm)": "{:.2f}",
            "Allowable (mm)": "{:.2f}"
        }),
        use_container_width=True
    )

    passing = df[df["Status"] == "PASS"]

    st.subheader("Preliminary Settlement-Based Breadth")

    if not passing.empty:

        optimum = passing.iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Smallest passing breadth",
            f"{optimum['Breadth (m)']:.2f} m"
        )

        c2.metric(
            "Total settlement",
            f"{optimum['Total settlement (mm)']:.2f} mm"
        )

        c3.metric(
            "Footing area",
            f"{optimum['Area (m²)']:.2f} m²"
        )

        st.success(
            "The selected breadth satisfies the specified "
            "settlement limit within the trial range."
        )

    else:

        st.error(
            "No trial breadth satisfies the specified settlement limit."
        )

    # Settlement graph

    st.subheader("Settlement versus Footing Breadth")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df["Breadth (m)"],
        df["Immediate (mm)"],
        marker="o",
        label="Immediate settlement"
    )

    ax.plot(
        df["Breadth (m)"],
        df["Primary (mm)"],
        marker="s",
        label="Primary consolidation"
    )

    ax.plot(
        df["Breadth (m)"],
        df["Secondary (mm)"],
        marker="^",
        label="Secondary consolidation"
    )

    ax.plot(
        df["Breadth (m)"],
        df["Total settlement (mm)"],
        marker="D",
        linewidth=2,
        label="Total settlement"
    )

    ax.axhline(
        settlement_limit,
        linestyle="--",
        label="Allowable settlement"
    )

    ax.set_xlabel("Footing breadth (m)")
    ax.set_ylabel("Settlement (mm)")
    ax.set_title("Settlement versus Footing Breadth")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    # Component graph

    st.subheader("Settlement Component Comparison")

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    x = np.arange(len(df))
    width = 0.25

    ax2.bar(
        x - width,
        df["Immediate (mm)"],
        width,
        label="Immediate"
    )

    ax2.bar(
        x,
        df["Primary (mm)"],
        width,
        label="Primary"
    )

    ax2.bar(
        x + width,
        df["Secondary (mm)"],
        width,
        label="Secondary"
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{b:.2f}" for b in df["Breadth (m)"]]
    )

    ax2.set_xlabel("Footing breadth (m)")
    ax2.set_ylabel("Settlement (mm)")
    ax2.set_title("Individual Settlement Components")
    ax2.grid(axis="y")
    ax2.legend()

    st.pyplot(fig2)

    # CSV export

    st.subheader("Download Results")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Settlement Results CSV",
        data=csv,
        file_name="settlement_results.csv",
        mime="text/csv"
    )

