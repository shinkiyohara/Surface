import numpy as np
import os
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.surface import Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json


def get_surface_sites(slab,tol, tag=True):
    """
    Returns the surface sites and their indices in a dictionary. The
    oriented unit cell of the slab will determine the coordination number
    of a typical site. We use VoronoiNN to determine the
    coordination number of bulk sites and slab sites. Due to the
    pathological error resulting from some surface sites in the
    VoronoiNN, we assume any site that has this error is a surface
    site as well. This will work for elemental systems only for now. Useful
    for analysis involving broken bonds and for finding adsorption sites.
        Args:
            tag (bool): Option to adds site attribute "is_surfsite" (bool)
                to all sites of slab. Defaults to False
        Returns:
            A dictionary grouping sites on top and bottom of the slab
            together.
            {"top": [sites with indices], "bottom": [sites with indices}
    TODO:
        Is there a way to determine site equivalence between sites in a slab
        and bulk system? This would allow us get the coordination number of
        a specific site for multi-elemental systems or systems with more
        than one unequivalent site. This will allow us to use this for
        compound systems.
    """

    from pymatgen.analysis.local_env import VoronoiNN

    # Get a dictionary of coordination numbers
    # for each distinct site in the structure
    a = SpacegroupAnalyzer(slab.oriented_unit_cell)
    ucell = a.get_symmetrized_structure()
    cn_dict = {}
    v = VoronoiNN(tol=tol)
    unique_indices = [equ[0] for equ in ucell.equivalent_indices]

    for i in unique_indices:
        el = ucell[i].species_string
        if el not in cn_dict.keys():
            cn_dict[el] = []
        # Since this will get the cn as a result of the weighted polyhedra, the
        # slightest difference in cn will indicate a different environment for a
        # species, eg. bond distance of each neighbor or neighbor species. The
        # decimal place to get some cn to be equal.
        cn = v.get_cn(ucell, i, use_weights=True)
        cn = float('%.5f' % (round(cn, 5)))
        if cn not in cn_dict[el]:
            cn_dict[el].append(cn)

    v = VoronoiNN(tol=tol)

    surf_sites_dict, properties = {"top": [], "bottom": []}, [],
    for i, site in enumerate(slab):
        # Determine if site is closer to the top or bottom of the slab
        top = site.frac_coords[2] > slab.center_of_mass[2]

        cn = float('%.5f' % (round(v.get_cn(slab, i, use_weights=True), 5)))
        try:
            # A site is a surface site, if its environment does
            # not fit the environment of other sites
            cn = float('%.5f' % (round(v.get_cn(slab, i, use_weights=True), 5)))
            if cn < min(cn_dict[site.species_string]):
                properties.append(True)
                key = "top" if top else "bottom"
                surf_sites_dict[key].append([site, i])
            else:
                properties.append(False)
        except RuntimeError:
            # or if pathological error is returned, indicating a surface site
            properties.append(True)
            key = "top" if top else "bottom"
            surf_sites_dict[key].append([site, i])

    if tag:
        slab.add_site_property("is_surf_site", properties)
    return surf_sites_dict

def ext_bulk_soap(matkey,bulkkeys):
    bulkkey=[i for i in bulkkeys if i in matkey][0]
    return bulkkey


orgdir='C:/Users/sin/RESEARCH/Informatics/project3/DATA/'

with open(orgdir+'structure_collection.json') as f:
    st_json=json.load(f)
    f.close()
with open(orgdir+'structure_collection_bulk.json') as f:
    st_bulk_json=json.load(f)
    f.close()
bulkkeys=list(st_bulk_json.keys())

surface_sites_dict={}
for matkey in st_json.keys():
    tmp_st=st_json[matkey]
    st = Structure.from_dict(tmp_st['st_after'])
    bulkkey=ext_bulk_soap(matkey,bulkkeys)
    st_bulk=Structure.from_dict(st_bulk_json[bulkkey])

    slab=Slab(lattice=st.lattice,
              species=st.species,
              coords=st.frac_coords,
              miller_index=[1,1,1],
              oriented_unit_cell=st_bulk,
              shift=0,
              scale_factor=1,
              reorient_lattice=False)

    print(slab.is_symmetric(),slab.have_equivalent_surfaces())

    surface_sites=[]
    surface_sites_prop=[]
    for tol in [0.0,0.25,0.5]:
        tmp_surface_sites=get_surface_sites(slab,tol=tol)
        surface_sites_prop.append(slab.site_properties['is_surf_site'])
    surface_sites_prop=np.asarray(surface_sites_prop)
    surface_sites_prop=np.any(surface_sites_prop,axis=0)
    surface_sites=np.arange(0,slab.num_sites,1)[surface_sites_prop]
    surface_sites= [ i for i in surface_sites if slab[i].frac_coords[2] > slab.center_of_mass[2]]  # top only
    surface_sites_top=np.zeros(slab.num_sites,dtype=bool)
    surface_sites_top[surface_sites]=True
    surface_sites_dict[matkey]=surface_sites_top.tolist()

with open(orgdir+'/surface_sites_top.json', 'w') as f:
    json.dump(surface_sites_dict, f, indent=4)
