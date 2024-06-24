"""
@author: qian rong
@date: 2023/10/27
tmap绘制分子结构树
"""


import pandas as pd
import tmap
from faerun import Faerun
from mhfp.encoder import MHFPEncoder
from rdkit.Chem import AllChem
from matplotlib.colors import ListedColormap

path = '/home/qianrong/final/src/parp1_vs/vs_parp1_smiles.csv'
# path = '/home/qianrong/final/src/parp1_vs/simi_2.csv'


df = pd.read_csv(path)
print(df.shape)
enc = MHFPEncoder()
lf = tmap.LSHForest(2048, 128)

##########################################################################################
"""
定义子结构
"""
# substruct_big = AllChem.MolFromSmiles("O=c1[nH]ncc2ccccc12")
# substruct_small = AllChem.MolFromSmiles("O=c1cccn[nH]1")
substruct_second = AllChem.MolFromSmiles('C(CCc([nH]1)nc2ccccc2c1=O)=O')


fingerprints = [tmap.VectorUint(enc.encode(s)) for s in df["smiles"]]
# has_sub1 = [AllChem.MolFromSmiles(smiles).HasSubstructMatch(substruct_big) for smiles in df['smiles']]
# has_sub2 = [AllChem.MolFromSmiles(smiles).HasSubstructMatch(substruct_small) for smiles in df['smiles']]
has_sub3 = [AllChem.MolFromSmiles(smiles).HasSubstructMatch(substruct_second) for smiles in df['smiles']]

# groups = df['category']
# labels_groups, groups = Faerun.create_categories(groups)

# labels_groups, groups = Faerun.create_categories(df['smiles'])


# Add the Fingerprints to the LSH Forest and index
lf.batch_add(fingerprints)
lf.index()
# Get the coordinates
x, y, s, t, _ = tmap.layout_from_lsh_forest(lf)

### 定义颜色
bin_cmap = ListedColormap(["#2ecc71", "#e74c3c"], name="bin_cmap")
custom_cmap = ListedColormap(["#2ecc71", "#9b59b6", "#e74c3c"], name="custom",)

# Now plot the data
faerun = Faerun(clear_color="#ffffff", view="front", coords=False)

# Add a scatter that is bigger than the one above, to add colored
# circles.
faerun.add_scatter(
    "generate_struc",
    {"x": x, "y": y, "c": has_sub3, "labels": df["smiles"]},
    colormap=bin_cmap,
    point_scale=7.5,
    max_point_size=100,
    shader="smoothCircle",
    has_legend=True,
    categorical=True,
    legend_title="Contains sub_structure",
    legend_labels=[(0, "No"), (1, "Yes")],
    interactive=False,
)


faerun.add_scatter(
    "generate_Basic_v5",
    {   "x": x,
        "y": y,
        "c": [
            #groups,
            # has_sub3,
            # has_sub1,
            # has_sub2,
            list(df['docking score'].values)
        ],
        "labels": df["smiles"]},
    point_scale=5,
    colormap = ["viridis"],    #rainbow
    has_legend=True,
    legend_title = 'docking score',
    categorical=[False],
    legend_labels=[
        # labels_groups
        #     [(0, "No"), (1, "Yes")],
            # [(0, "No"), (1, "Yes")],
            # labels_groups
        ],
    series_title=[
        # 'sub1',
        # 'sub2',
        # 'smiles category'
        # 'substructure',
        'docking score'
    ],
    shader = 'smoothCircle'
)

faerun.add_tree("generate_tree", {"from": s, "to": t}, point_helper="generate_Basic_v5")

# Choose the "smiles" template to display structure on hover
faerun.plot('generate_Basic_v5', template="smiles", notebook_height=750)
