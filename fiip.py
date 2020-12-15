
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
from scipy import signal
import matplotlib.pyplot as plt
from DataFeeder import PreTrainDiscrFeeder
from nets import FIIPDiscriminator

datajson = "/home/turro/Repos/FIIP/processingmetadata.json"
with open(datajson) as jsonfh:
    metadata = json.load(jsonfh)

def rawtoSTFT(indir,outdir):
    targetsamplingrate = metadata["targetsamplerate"]
    dT = metadata["netdT"]
    for filename in os.listdir(indir):
        if filename.endswith((".wav",".WAV")):
            wav = os.path.join(indir,filename)
            rawdata,samplingrate = sf.read(wav)
            print(rawdata.shape)
            rawdata = rawdata[::int(samplingrate/targetsamplingrate)][:,0]
            print(rawdata.shape)
            f, t, Zxx = signal.stft(rawdata, targetsamplingrate, 
                                    nperseg=int(targetsamplingrate*dT)
                                    )
            print(f.shape,t.shape,Zxx.shape)
            data = {
                "sr":targetsamplingrate,
                "nperseg": int(targetsamplingrate*dT),
                "f":f,
                "t":t,
                "Z":Zxx
            }
            with open(os.path.join(outdir,"{}.pkl".format(filename.split(".")[0])),"wb+") as fh:
                pickle.dump(data,fh)

def STFTtoslices(indir,outdir):
    for filename in os.listdir(indir):
        with open(os.path.join(indir,filename),"rb")  as fh:
            data = pickle.load(fh)
        for sliceidx,t in enumerate(data["t"]):
            if t > metadata["slicesize"]:
                break
        padding = np.zeros((data["Z"].shape[0],sliceidx - data["Z"].shape[1]%sliceidx))
        tocut = np.concatenate((data["Z"],padding),axis=1)

        for i in range(0,tocut.shape[1],sliceidx):
            chunk = tocut[:,i :  i + sliceidx]
            with open(os.path.join(outdir,str(int(i/sliceidx)) + filename),"wb+") as fh:
                freqsplit = np.stack((chunk.real,chunk.imag),axis = 2)
                print(freqsplit.shape)
                pickle.dump(freqsplit,fh)

def badrawtoSTFT():
    badsounddir = os.path.join(metadata["basedir"],metadata["badfreqdir"])
    badsoundtimechunkdir = os.path.join(metadata["basedir"],metadata["badactortimechunks"])
    for actor in metadata["actors"]:
        rawtoSTFT(os.path.join(badsounddir,actor),os.path.join(badsoundtimechunkdir,actor))

def badSTFTtoslices():
    badtimechunkdir = os.path.join(metadata["basedir"],metadata["badactortimechunks"])
    badfreqchunkdir = os.path.join(metadata["basedir"],metadata["badcurated"])
    for actor in metadata["actors"]:
        STFTtoslices(os.path.join(badtimechunkdir,actor),os.path.join(badfreqchunkdir,actor))

def generalrawtoSTFT():
    chunkduration = metadata["netdT"]
    generalcorpusdir = os.path.join(metadata["basedir"],metadata["rawcorpusesdir"])
    generaltimechunkdir = os.path.join(metadata["basedir"],metadata["generaltimechunks"])
    rawtoSTFT(generalcorpusdir,generaltimechunkdir)


def generalSTFTtoslices():
    generaltimechunkdir = os.path.join(metadata["basedir"],metadata["generaltimechunks"])
    generalfreqchunkdir = os.path.join(metadata["basedir"],metadata["generalcurated"])
    STFTtoslices(generaltimechunkdir,generalfreqchunkdir)

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
    data = pickle.load(open("/media/turro/CHARLES/fiip/data/netfeeder/pretraindiscriminator/training/good/33SBC001.pkl","rb"))
    def adder(a):
        return 100*(a[0] + a[1])
    data = np.apply_along_axis(adder,2,data)
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()

    print(data.shape)


def pretraindiscriminator():
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    targetsamplingrate = metadata["targetsamplerate"]
    chunkduration = float(metadata["netdT"])

    traininggooddatadir = os.path.join(metadata["basedir"],metadata["gooddatalocations"][0][0])
    trainingbaddatadir = os.path.join(metadata["basedir"],metadata["baddatalocations"][0][0])
    validationgooddatadir = os.path.join(metadata["basedir"],metadata["gooddatalocations"][1][0])
    validationbaddatadir = os.path.join(metadata["basedir"],metadata["baddatalocations"][1][0])
    preTrainDiscDataset = PreTrainDiscrFeeder(traininggooddatadir,trainingbaddatadir,(552,41,2))
    preValidDiscDataset = PreTrainDiscrFeeder(validationgooddatadir,validationbaddatadir,(552,41,2))

    discriminator = FIIPDiscriminator((552,41,2))
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
            "generalrawtoSTFT":generalrawtoSTFT,
            "badrawtoSTFT":badrawtoSTFT,

            "generalSTFTtoslices":generalSTFTtoslices,
            "badSTFTtoslices":badSTFTtoslices,

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
