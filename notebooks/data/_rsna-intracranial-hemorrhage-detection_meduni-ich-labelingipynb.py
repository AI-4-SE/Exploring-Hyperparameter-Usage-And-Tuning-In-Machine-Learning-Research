#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pydicom kornia opencv-python scikit-image pyarrow')


# In[ ]:


from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.medical.imaging import *

import pydicom

seed = 42  # random number generator seed => macht Zufall wiederholbar


# In[ ]:


labels = {}   # hier werden manuell gelabelte Beispiele gespeichert
verlauf = []  # zur Überprüfung etwaiger Fehler


# # Daten anschauen

# In[ ]:


path_data = Path("../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/")
assert path_data.exists()


# In[ ]:


# Fastai überprüft normalerweise ob die Dateien wirklich korrekt sind
# das macht das Laden der DICOMs ziemlich langsam
# zudem verwenden wir hier ein professionell erstelltes Datenset
# daher beschleunigen wir das Laden der Daten indem wir hier die eingebauten Funktionen von fastai durch unsere eigenen ersetzen
# zudem bauen wir ein Limit ein falls wir nicht alle DICOMs laden wollen

def get_files(path, extensions=None, folders=None, followlinks=True, limit=None):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    path = Path(path)
    folders=L(folders)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}    
    
#   f = [o.name for o in os.scandir(path) if o.is_file()]  # hier überprüft das Original ob es wirklich Dateien sind
    f = [o.name for o in os.scandir(path)]
    if limit:
        f = f[:limit]
            
    res = _get_files(path, f, extensions)
    
    return L(res)


def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res


def get_dicom_files(path, folders=None, limit=None):
    "Get dicom files in `path` recursively, only in `folders`, if specified."
    return get_files(path, extensions=[".dcm",".dicom"], folders=folders, limit=limit)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# das laden aller DICOMs dauert trotzdem etwas\ndicoms = get_dicom_files(path_data, limit=None)\n')


# In[ ]:


dicoms


# In[ ]:


# hier speichern wir DICOMS die wir schon mal gesehen haben damit wir nicht eines doppelt erwischen
seen_dicoms = set()


# In[ ]:


# diese Funktion wählt jedes Mal eine neue zufällige DICOM Datei aus
def get_random_dicom(dicoms):
    dcm = random.choice(dicoms)
    
    while dcm in seen_dicoms:
        # neues random dicom holen
        dcm = random.sample(dicoms)
    
    seen_dicoms.add(dcm)   
    return dcm


# ## Datenset ausbalanzieren
# im ursprünglichen Datenset sind nur ca 5% der Bilder von Gehirnblutungen  
# hier gleichen wir das aus indem wir genug nicht-Blutungen raus filtern bis wir ein 50:50 Verhältnis haben  
# (Downsampling)

# In[ ]:


path_csv = Path("../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv")

df = pd.read_csv(path_csv)


# In[ ]:


df[["filename", "exact_label"]] = df["ID"].str.rsplit("_", 1, expand=True)
df.set_index("filename", inplace=True)


# In[ ]:


df_0 = (df[df["Label"] == 0])
df_1 = (df[df["Label"] == 1])

n = len(df_1)


# In[ ]:


samples = pd.concat([df_1, df_0.sample(n=n, random_state=seed)])


# In[ ]:


list(samples["Label"].value_counts().items())


# In[ ]:


print("Blutung ja/nein:    Anzahl DICOMs")
for k, v in samples["Label"].value_counts().items():
    print(f" {k}                   {v}")


# ## Dicom zufällig auswählen und labeln

# In[ ]:


scales = [
    dicom_windows.brain,  # W:80, L:40
#     True,  # normalized
#     dicom_windows.subdural,  # W:254, L:100
]

titles = [
    "Brain Window",
#     "Normalized",
#     "Subdural Window"
]


# ## Daten labeln

# In[ ]:


# hier holen wir uns ein neues zufälliges DICOM
sample = get_random_dicom(dicoms)
sample.name


# In[ ]:


# DICOM anzeigen
bild_größe = 12

for scale, axis, title in zip(scales, subplots(len(scales),1,imsize=bild_größe)[1].flat, titles):
    dcm = sample.dcmread()
    dcm.show(scale=scale, ax=axis, title=title)


# In[ ]:


# ist eine Blutung
labels[sample.name] = 1
verlauf.append(f"{sample.name}=1 ")


# In[ ]:


# ist KEINE Blutung
labels[sample.name] = 0
verlauf.append(f"{sample.name}=0 ")


# In[ ]:


print(f"Bisher gelabelt: {len(labels)} DICOMs")


# In[ ]:


labels


# In[ ]:


verlauf[-5:]


# ## Labels speichern

# In[ ]:


# Labels speichern

name_student = ""
assert name_student, "Bitte einen gültigen Namen ausfüllen"

pd.Series(labels).to_csv(f"./{name_student}.csv", header=False)

print("Nach dem Speichern nicht vergessen die Daten auch runter zu laden!")


# In[ ]:




