# DISTROI

**DISTROI**, or **D**isk **I**nner **ST**ructure **R**econnaissance through **O**ptical **I**nteferometry, is a Python 
package intended to compute optical interferometry (OI) observables  from radiative transfer (RT) model images and compare 
them to observations from modern OI observatories (VLTI, CHARA, ...).

# Dependencies

DISTROI depends on a few other Python packages. The main dependencies are listed in the requirements.txt file supplied 
with the source code. In addition,  the distroiEnv.yml file can be used to create a conda environment with all 
dependencies installed via 'conda env create -f distroiEnv.yml'.

# Examples

After installing all the dependencies, run the single_disk_model.py script in the 'examples' folder. This will
run OI observable calculation for an example model of the dusty circumbinary disk surroudning the IRAS 08544-4431
system. This model is calculated using the MCFOST RT code (https://ipag.osug.fr/~pintec/mcfost/docs/html/overview.html).
DISTROI is applied to four separate datasets, covering the H, K, L and N-band, and shows the resulting plots.

# License

Copyright 2024 Toon De Prins

DISTROI is free software: you can redistribute it and/or modify  it under the terms of the GNU General Public License 
as published by  the Free Software Foundation, either version 3.0 of the License, or (at your option) any later version.
DISTROI is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the  GNU General Public License for more details. 
You should have received a copy of the GNU General Public License along with DISTROI. If not, see 
<https://www.gnu.org/licenses/>.

---
**NOTE**

This project is under active development. Its current structure is bound to change significantly in the near future.
---
