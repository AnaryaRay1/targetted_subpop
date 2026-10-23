def alpha_21_constraint(reference_params, alpha11):
    return dict(
        minimum=reference_params["minimum"],
        maximum=alpha11
    )

def alpha_3_constraint(reference_params, alpha11):
    return dict(
        minimum=-alpha11,
        maximum=reference_params["maximum"]
    )

def m_break_condition(reference_params, mmmax2):
    return dict(
        minimum=reference_params["minimum"],
        maximum=mmax2
        )
def m_break_condition_restricted(reference_params, mmax1):
    return dict(
        minimum=reference_params["minimum"],
        maximum=min(mmax1, reference_params["maximum"])
        )

# def mmax2_condition(reference_params, mmax1):
#     return dict(peak = mmax1
#             )
    
def chi_eff_min_condition(reference_params, chi_eff_max3):
    return dict(
        minimum=reference_params["minimum"],
        maximum=chi_eff_max3
        )

def m2min_condition(reference_params, mmin1):
    return dict(
        minimum=reference_params["minimum"],
        maximum=mmin1
        )

def mmin3_condition(reference_params, mmin1, m2min):
    return dict(
        minimum = mmin1 + m2min,
        maximum = 2.0 * mmin1,
    )

def beta_chip3_condition(reference_params, alpha_chip3_1m):
    return dict(
        minimum = reference_params["minimum"],
        maximum = 1.0 + alpha_chip3_1m,
    )