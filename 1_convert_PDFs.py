# =============================================================================
# Loading packages
# =============================================================================

import os

# =============================================================================
# Import functions - UPDATE IF FILE NAME CHANGES
# =============================================================================

os.chdir(r'W:\your_folder\Python')
from functions import *

# =============================================================================
# Save protected PDFs as unprotected PDFs
# =============================================================================

# Cancellation
lotSaveNew(r'W:\your_folder\sorted data\Cancellation',
                r'W:\your_folder\data - unprotected\Cancellation')             

# Contribution break
lotSaveNew(r'W:\your_folder\sorted data\Contribution_break',
                r'W:\your_folder\data - unprotected\Contribution_break')

# Premium waiver
lotSaveNew(r'W:\your_folder\sorted data\Premium_waiver',
                r'W:\your_folder\data - unprotected\Premium_waiver')

# Increase
lotSaveNew(r'W:\your_folder\sorted data\Increase',
                r'W:\your_folder\data - unprotected\Increase')

# Policyholder change
lotSaveNew(r'W:\your_folder\sorted data\Policyholder_change',
                r'W:\your_folder\data - unprotected\Policyholder_change')
