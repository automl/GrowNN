import datetime
import traceback

name = "SMAC3"
package_name = "smac"
author = (
    "\t### ###, ### ###, ### ###, ######, "
    "### ###,\n\t### ###, ###, ### ### and ### ###"
)

author_email = "fh@cs.uni-freiburg.de"
description = "SMAC3, a Python implementation of 'Sequential Model-based Algorithm Configuration'."
url = "https://www.automl.org/"
project_urls = {
    "Documentation": "https://https://github.com/automl.github.io/SMAC3/main",
    "Source Code": "https://github.com/https://github.com/automl/smac",
}
copyright = f"""
    Copyright {datetime.date.today().strftime('%Y')}, ### ###, ### ###,
    ### ###, ######, ### ###, ### ###, ###, ### ###
    and ### ###"""
version = "2.1.0"


try:
    from smac.callback.callback import Callback
    from smac.facade import (
        AlgorithmConfigurationFacade,
        BlackBoxFacade,
        HyperbandFacade,
        HyperparameterOptimizationFacade,
        MultiFidelityFacade,
        RandomFacade,
    )
    from smac.runhistory.runhistory import RunHistory
    from smac.scenario import Scenario

    __all__ = [
        "Scenario",
        "RunHistory",
        "BlackBoxFacade",
        "HyperparameterOptimizationFacade",
        "MultiFidelityFacade",
        "AlgorithmConfigurationFacade",
        "RandomFacade",
        "HyperbandFacade",
        "Callback",
    ]
except ModuleNotFoundError as e:
    print(e)
    traceback.print_exc()
