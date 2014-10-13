import numpy as np
# some physics
# mass of particles
mass = {
    'tau': 1.77682,
    'lep': 0,
    'jet_leading': 4.190,
    'jet_subleading': 4.190
}

# this a a set of particles
class PUnit:
    # pt, eta, phi are vectors
    # set -999 to nan first
    def __init__(self, name, pt, phi, eta = None, m = 0):
        self.name = name
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.p_x = pt * np.cos(phi)
        self.p_y = pt * np.sin(phi)
        self.m = m
        self.E_tri = np.sqrt(self.p_x**2 + self.p_y**2 + m)
        if eta != None:
            self.p_z = pt * np.sinh(eta)
            self.p_mo = pt * np.cosh(eta)
            self.E_inv = np.sqrt(self.p_x**2 + self.p_y**2 + self.p_z**2 + m)

# set of PUnit, sum statistics
class PSet:
    def __init__(self, pset):
        self.names = set(a.name for a in pset)
        self.p_x = sum(a.p_x for a in pset)
        self.p_y = sum(a.p_y for a in pset)
        self.E_tri = sum(a.E_tri for a in pset)
        self.pts = np.sqrt(self.p_x**2 + self.p_y**2)
        # this is m_tri square
        self.m_tri = np.sqrt(self.E_tri**2 - self.p_x**2 - self.p_y**2)

        if len([a for a in pset if a.eta is None]) == 0:
            self.p_z = sum(a.p_z for a in pset)
            self.p_mo = sum(a.p_mo for a in pset)
            self.E_inv = sum(a.E_inv for a in pset)
            self.m_inv = np.sqrt(self.E_inv**2 - self.p_x**2 -
                                 self.p_y**2 - self.p_z**2)
            self.pnorm = np.sqrt(self.p_x**2 + self.p_y**2 +self.p_z**2)
            self.pt_xz = np.sqrt(self.p_x**2 + self.p_z**2)
            self.pt_yz = np.sqrt(self.p_y**2 + self.p_z**2)
        else:
            self.p_z = None
    # get list of features
    def get_feats(self):
        fmap = {}
        nm = '_'.join(sorted(self.names))
        fmap[nm+'.px'] = self.p_x
        fmap[nm+'.py'] = self.p_y
        fmap[nm+'.pts'] = self.pts
        fmap[nm+'.E_tri'] = self.E_tri
        fmap[nm+'.m_tri'] = self.m_tri
        if self.p_z != None:
            fmap[nm+'.p_z'] = self.p_z
            fmap[nm+'.E_inv'] = self.E_inv
            fmap[nm+'.m_inv'] = self.m_inv
            fmap[nm+'.pnorm'] = self.pnorm
            fmap[nm+'.pt_xz'] = self.pt_xz
            fmap[nm+'.pt_yz'] = self.pt_yz
        return fmap
    # number of units
    def nunits(self):
        return len(self.names)

def filter_pri(data, header):
    data = data.copy()
    flst = []
    for i in range(len(header)):
        nm = header[i]
        # remove absolute angle, of tau, lep, jet
        if nm.rsplit('_')[-1] == 'eta':
            continue
        if nm.rsplit('_')[-1] == 'phi':
            continue
        # pt is already part of set feature
        if nm.rsplit('_')[-1] == 'pt':
            continue
        flst.append(i)
    return data[:,flst]            

def make_punits(data, header, mass = {}):
    data = data.copy()
    data[data == -999] = np.nan
    fmap = {}
    for i in range(len(header)):
        fmap[header[i].strip()] = i
    ret = []
    for nm in ['tau', 'lep', 'jet_leading', 'jet_subleading']:
        if nm in mass:
            m = mass[nm]
        else:
            m = 0
        ret.append(PUnit(name = nm,
                         pt = data[:,fmap['PRI_'+nm+'_pt']],
                         phi= data[:,fmap['PRI_'+nm+'_phi']],
                         eta = data[:,fmap['PRI_'+nm+'_eta']],
                         m = m))
    ret.append(PUnit(name = 'met',
                     pt = data[:,fmap['PRI_met_sumet']],
                     phi= data[:,fmap['PRI_met_phi']],
                     eta= None))
    return ret


def make_psets(punit, plst, ret):
    if len(punit) == 0:
        if len(plst) != 0: 
            ret.append(PSet(plst))
    else:
        make_psets(punit[1:], plst, ret)
        make_psets(punit[1:], plst+[punit[0]], ret)
    
def load_train(fname, use_mass = False, filter_angle = False):
    ### load data in do training
    train = np.loadtxt(fname, delimiter=',', skiprows=1, 
                       converters={32: lambda x:int(x=='s'.encode('utf-8')) } )
    label  = train[:,32]
    data   = train[:,1:31]
    weight = train[:,31]
    if use_mass:
        punits = make_punits(data, open(fname).readline().split(',')[1:31], mass)
    else:
        punits = make_punits(data, open(fname).readline().split(',')[1:31])
    print 'train: punits are %s' % str([a.name for a in punits])
    psets = []
    make_psets(punits, [], psets)
    print '%d psets discovered' % len(psets)
    if filter_angle:
        data = filter_pri(data, open(fname).readline().split(',')[1:31])
    return label, data, weight, punits, psets

def load_test(fname, use_mass = False, filter_angle = False):
    ### load data in do training
    test = np.loadtxt(fname, delimiter=',', skiprows=1 )
    idx = test[:,0]
    data = test[:,1:31]
    if use_mass:
        punits = make_punits(data, open(fname).readline().split(',')[1:31], mass)
    else:
        punits = make_punits(data, open(fname).readline().split(',')[1:31])    
    print 'test: punits are %s' % str([a.name for a in punits])
    psets = []
    make_psets(punits, [], psets)
    print '%d psets discovered' % len(psets)
    if filter_angle:
        data = filter_pri(data, open(fname).readline().split(',')[1:31])
    return idx, data, punits, psets

def mkf_pset(pset, features):
    fmap = {}
    for p in pset:
        for k, v in p.get_feats().items():            
            if k.split('.')[1] in features:
                assert k not in fmap
                fmap[k] = v

    # sort, so we are consistent
    feat_list = [ x[1] for x in sorted(fmap.items(), key = lambda x:x[0]) ]
    ret = np.vstack(feat_list).T
    ret[np.isnan(ret)] = -999
    return ret

# dot product of two momentums of two pset
def make_dot(a, b):
    assert len(a.names.intersection(b.names)) == 0
    name_a = '_'.join(sorted(a.names))
    name_b = '_'.join(sorted(b.names))
    if name_a > name_b:
        name_a, name_b = name_b, name_a
    nm = name_a + '*' + name_b

    fmap = {}
    tvalue = a.p_x * b.p_x + a.p_y * b.p_y
    tdiff = (a.p_x - b.p_x)**2 + (a.p_y - b.p_y)**2

    fmap[nm+".dot_tri"] = tvalue
    fmap[nm+".pdiff_tri"] = tdiff
    fmap[nm+".cos_tri"] = tvalue / (a.pts * b.pts)

    if a.p_z != None and b.p_z != None:
        dvalue = a.p_x*b.p_x + a.p_y*b.p_y +a.p_z*b.p_z
        fmap[nm+".pdiff_inv"] = tdiff + (a.p_z - b.p_z)**2
        fmap[nm+".dot_inv"] = dvalue
        fmap[nm+".cos_inv"] = dvalue / (a.pnorm * b.pnorm)    
    return fmap

def make_cross(a, b):
    assert len(a.names.intersection(b.names)) == 0
    if a.p_z == None or b.p_z == None:
        return {}
    name_a = '_'.join(sorted(a.names))
    name_b = '_'.join(sorted(b.names))
    if name_a > name_b:
        name_a, name_b = name_b, name_a
        a, b = b, a
    nm = name_a + '*' + name_b
    fmap = {}
    # cross product
    fmap[nm+".cross_x"] = a.p_y * b.p_z - a.p_z * b.p_y
    fmap[nm+".cross_y"] = a.p_z * b.p_x - a.p_x * b.p_z
    fmap[nm+".cross_z"] = a.p_x * b.p_y - a.p_y * b.p_x
    dvalue = a.pnorm * b.pnorm
    # unit vector
    fmap[nm+".cross_e_x"] = fmap[nm+'.cross_x'] / dvalue
    fmap[nm+".cross_e_y"] = fmap[nm+'.cross_y'] / dvalue
    fmap[nm+".cross_e_z"] = fmap[nm+'.cross_z'] / dvalue    
    return fmap

def mkf_dotcross(pset, features):
    fmap = {}
    for i in range(len(pset)-1):
        for j in range(i+1, len(pset)):
            a = pset[i]; b = pset[j]
            if len(a.names.intersection(b.names)) == 0:
                for k, v in make_dot(a,b).items():
                    if k.split('.')[1] in features:
                        assert k not in fmap
                        fmap[k] = v
                for k, v in make_cross(a,b).items():
                    if k.split('.')[1] in features:
                        assert k not in fmap
                        fmap[k] = v
    # sort, so we are consistent
    feat_list = [ x[1] for x in sorted(fmap.items(), key = lambda x:x[0]) ]
    ret = np.vstack(feat_list).T
    ret[np.isnan(ret)] = -999
    return ret

def make_det(a,b,c):
    assert len(a.names.intersection(b.names)) == 0
    assert len(b.names.intersection(c.names)) == 0
    assert len(c.names.intersection(a.names)) == 0
    if a.p_z == None or b.p_z == None or c.p_z == None:
        return {}    
    name_a = '_'.join(sorted(a.names))
    name_b = '_'.join(sorted(b.names))
    name_c = '_'.join(sorted(c.names))
    lst = sorted( [ (name_a, a), (name_b, b), (name_c, c)], key = lambda x:x[0])
    name_a, a = lst[0]
    name_b, b = lst[1]
    name_c, c = lst[2]    
    nm = name_a+'*'+name_b+'*'+name_c
    # determinant
    vol = c.p_x * (a.p_y * b.p_z - a.p_z * b.p_y) +\
        c.p_y * (a.p_z * b.p_x - a.p_x * b.p_z) +\
        c.p_z * (a.p_x * b.p_y - a.p_y * b.p_x)
    
    fmap = {}
    fmap[nm+'.det'] = vol
    fmap[nm+'.det_norm']  = vol / (a.pnorm * b.pnorm * c.pnorm)
    fmap[nm+'.det_ab']  = vol / (a.pnorm * b.pnorm)
    fmap[nm+'.det_ac']  = vol / (a.pnorm * c.pnorm)
    fmap[nm+'.det_bc']  = vol / (b.pnorm * c.pnorm)
    return fmap    

def mkf_det(pset, features):
    fmap = {}
    for i in range(len(pset)-2):
        for j in range(i+1, len(pset)-1):
            for k in range(j+1, len(pset)):
                a = pset[i]; b = pset[j]; c = pset[k]
                if len(a.names.intersection(b.names)) != 0: continue
                if len(b.names.intersection(c.names)) != 0: continue
                if len(c.names.intersection(a.names)) != 0: continue
                for k, v in make_det(a,b,c).items():
                    if k.split('.')[1] in features:
                        assert k not in fmap
                        fmap[k] = v
    # sort, so we are consistent
    feat_list = [ x[1] for x in sorted(fmap.items(), key = lambda x:x[0]) ]
    ret = np.vstack(feat_list).T
    ret[np.isnan(ret)] = -999
    return ret

