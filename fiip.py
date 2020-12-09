
import soundfile as sf
from scipy.fft import fft,ifft
import json
import os
import pickle
import numpy as np
import random
import shutil
import pathlib
import keras
import tensorflow as tf
from DataFeeder import PreTrainDiscrFeeder
from nets import FIIPDiscriminator

datajson = "/home/turro/Repos/FIIP/processingmetadata.json"
targetsamplingrate = 22050
with open(datajson) as jsonfh:
    metadata = json.load(jsonfh)

def rawtochunks(indir,outdir,chunkduration,tag):
    rawwavs = []
    for filename in os.listdir(indir):
        if filename.endswith((".wav",".WAV")):
            rawwavs.append(os.path.join(indir,filename))
    idx = 0
    for wav in rawwavs:
        rawdata,samplingrate = sf.read(wav)
        chunks = []
        chunkstep = int(chunkduration*samplingrate)
        for chunkidx in range(0,len(rawdata) - chunkstep,chunkstep):
            for band in range(rawdata.shape[1]):
                chunks.append(rawdata[chunkidx : chunkidx + chunkstep,band]) #[band])
        for chunk in chunks:
            if np.average(chunk) != 0:
                with open(os.path.join(outdir,"{}{}.pkl".format(tag,str(idx).zfill(6))),"wb+") as fh:
                    pickle.dump(chunk[::int(samplingrate/targetsamplingrate)][:int(targetsamplingrate*chunkduration)],fh)
                idx += 1

def chunktofreq(indir,outdir):
    for filename in os.listdir(indir):
        with open(os.path.join(indir,filename),"rb")  as fh:
            chunk = pickle.load(fh)
        with open(os.path.join(outdir,filename),"wb+") as fh:
            freq = fft(chunk)
            freqsplit = np.stack((freq.real,freq.imag))
            pickle.dump(freqsplit,fh)

def badsoundtochunks():
    chunkduration = metadata["netdT"]
    badsounddir = os.path.join(metadata["basedir"],metadata["badfreqdir"])
    badsoundtimechunkdir = os.path.join(metadata["basedir"],metadata["badactortimechunks"])
    for actor in metadata["actors"]:
        rawtochunks(os.path.join(badsounddir,actor),os.path.join(badsoundtimechunkdir,actor),chunkduration,actor)

def badsoundtofreq():
    badtimechunkdir = os.path.join(metadata["basedir"],metadata["badactortimechunks"])
    badfreqchunkdir = os.path.join(metadata["basedir"],metadata["badcurated"])
    for actor in metadata["actors"]:
        chunktofreq(os.path.join(badtimechunkdir,actor),os.path.join(badfreqchunkdir,actor))

def generalrawtochunks():
    chunkduration = metadata["netdT"]
    generalcorpusdir = os.path.join(metadata["basedir"],metadata["rawcorpusesdir"])
    generaltimechunkdir = os.path.join(metadata["basedir"],metadata["generaltimechunks"])
    rawtochunks(generalcorpusdir,generaltimechunkdir,chunkduration,"general")


def generalchunkstofreq():
    generaltimechunkdir = os.path.join(metadata["basedir"],metadata["generaltimechunks"])
    generalfreqchunkdir = os.path.join(metadata["basedir"],metadata["generalcurated"])
    chunktofreq(generaltimechunkdir,generalfreqchunkdir)

def reconstructfreqdir():
    dirtoreconstruct = os.path.join(metadata["basedir"],metadata["rebuild"])
    fulllist = [os.path.join(dirtoreconstruct,f) for f in os.listdir(dirtoreconstruct)][::3]
    timeconvert = []
    for filename in fulllist:
        with open(filename,"rb") as fh:
            data = pickle.load(fh)
        timeconvert.append(ifft(data))
    soundfile = np.concatenate(timeconvert)
    sf.write("test.wav",soundfile.real,targetsamplingrate)

def generatenetfeederstructure():
    baddatalocations = [os.path.join(metadata["basedir"],d[0]) for d in metadata["baddatalocations"]]
    gooddatalocations = [os.path.join(metadata["basedir"],d[0]) for d in metadata["gooddatalocations"]]
    alldirstomake = baddatalocations + gooddatalocations
    for dirtomake  in alldirstomake:
        pathlib.Path(dirtomake).mkdir(parents=True, exist_ok=True)
def splitintotrainingsets():
    actortomodel = "homestead"

    gooddatadir = os.path.join(metadata["basedir"],metadata["generalcurated"])
    gooddatadirrep = os.path.join(metadata["basedir"],metadata["goodcurated"],actortomodel)
    baddatadir = os.path.join(metadata["basedir"],metadata["badcurated"],actortomodel)

    baddatalocations = [os.path.join(metadata["basedir"],d[0]) for d in metadata["baddatalocations"]]
    baddataratio = [d[1] for d in metadata["baddatalocations"]]
    gooddatalocations = [os.path.join(metadata["basedir"],d[0]) for d in metadata["gooddatalocations"]]
    gooddataratio = [d[1] for d in metadata["gooddatalocations"]]


    goodslices = [os.path.join(gooddatadir,d) for d in os.listdir(gooddatadir)]
    goodslicesrep =  [os.path.join(gooddatadirrep,d) for d in os.listdir(gooddatadirrep)]
    for i in range(int(metadata["actorrepeatcnt"])):
        goodslices.extend(goodslicesrep)
    badslices = [os.path.join(baddatadir,d) for d in os.listdir(baddatadir)]
    random.shuffle(goodslices)
    random.shuffle(badslices)

    placeidx = 0
    idx = 0
    for slicetomove in goodslices:
        shutil.move(slicetomove,gooddatalocations[placeidx])
        idx += 1
        if idx == gooddataratio[placeidx]:
            idx = 0
            placeidx = (placeidx + 1)%len(gooddataratio)

    placeidx = 0
    idx = 0
    for slicetomove in badslices:
        shutil.move(slicetomove,baddatalocations[placeidx])
        idx += 1
        if idx == baddataratio[placeidx]:
            idx = 0
            placeidx = (placeidx + 1)%len(baddataratio)

def sandbox():
    chunkduration = float(metadata["netdT"])

    traininggooddatadir = os.path.join(metadata["basedir"],metadata["gooddatalocations"][0][0])
    trainingbaddatadir = os.path.join(metadata["basedir"],metadata["baddatalocations"][0][0])
    validationgooddatadir = os.path.join(metadata["basedir"],metadata["gooddatalocations"][1][0])
    validationbaddatadir = os.path.join(metadata["basedir"],metadata["baddatalocations"][1][0])
    preTrainDiscDataset = PreTrainDiscrFeeder(traininggooddatadir,trainingbaddatadir,(int(chunkduration*targetsamplingrate),2))
    preValidDiscDataset = PreTrainDiscrFeeder(validationgooddatadir,validationbaddatadir,(int(chunkduration*targetsamplingrate),2))

    discriminator = FIIPDiscriminator()
    print(discriminator.summary())
    for  i in range(5):
        batch = preTrainDiscDataset.__getitem__(i)
        print(batch[0].shape,batch[1].shape)
    preTrainDiscDataset.on_epoch_end()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    for  i in range(5):
        batch = preTrainDiscDataset.__getitem__(i)
        print(batch[0].shape,batch[1].shape)
def pretraindiscriminator():
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    chunkduration = float(metadata["netdT"])

    traininggooddatadir = os.path.join(metadata["basedir"],metadata["gooddatalocations"][0][0])
    trainingbaddatadir = os.path.join(metadata["basedir"],metadata["baddatalocations"][0][0])
    validationgooddatadir = os.path.join(metadata["basedir"],metadata["gooddatalocations"][1][0])
    validationbaddatadir = os.path.join(metadata["basedir"],metadata["baddatalocations"][1][0])
    preTrainDiscDataset = PreTrainDiscrFeeder(traininggooddatadir,trainingbaddatadir,(int(chunkduration*targetsamplingrate),2))
    preValidDiscDataset = PreTrainDiscrFeeder(validationgooddatadir,validationbaddatadir,(int(chunkduration*targetsamplingrate),2))

    discriminator = FIIPDiscriminator()
    print(discriminator.summary())
    discriminator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives()
        ]
    )

    hist = discriminator.fit(
        x = preTrainDiscDataset,
        epochs=50,
        verbose=1,
        validation_data=preValidDiscDataset
    )
    with open(os.path.join(metadata["basedir"],metadata["lasttraininghistory"]),"wb+") as fh:
        pickle.dump(hist.history,fh)
if __name__ == "__main__":
        jobs = {
            "generalrawtochunks":generalrawtochunks,
            "generalchunkstofreq":generalchunkstofreq,
            "badrawtochunks":badsoundtochunks,
            "badsoundtofreq":badsoundtofreq,
            "reconstructfreqdir":reconstructfreqdir,
            "splitintotrainingsets": splitintotrainingsets,
            "generatenetfeederstructure": generatenetfeederstructure,
            "pretraindiscriminator": pretraindiscriminator,
            "sandbox":sandbox,
            "exit":exit
        }
        print("jobs")
        jobidx = []
        for i,j in enumerate(jobs):
            print(i, ": ", j)
            jobidx.append(jobs[j])
             
        while True:
            job = int(input("Select a job index: "))
            jobidx[job]()
