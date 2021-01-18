
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
from DataFeeder import PreTrainDiscrFeeder, GeneratorFeeder
from nets import FIIPDiscriminator, FIIPGenerator

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
        cutidxs = []
        lastT = 0
        for i,t in enumerate(data["t"]):
            if t - lastT > metadata["slicesize"]:
                lastT = t
                cutidxs.append(i)

        for i in range(len(cutidxs) - 1):
            chunk = data["Z"][:metadata["freqidxcut"],cutidxs[i] :  cutidxs[i + 1] ]
            with open(os.path.join(outdir,str(i) + filename),"wb+") as fh:
                freqsplit = np.absolute(chunk)
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
    actortomodel = "homesteadcurated"

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
    trainingbaddatadir   = os.path.join(metadata["basedir"],metadata["baddatalocations"][0][0])
    validationbaddatadir = os.path.join(metadata["basedir"],metadata["baddatalocations"][1][0])
    freqidxcut = metadata["freqidxcut"]
    preTrainDataset = GeneratorFeeder(trainingbaddatadir,(freqidxcut,41,1))
    for i in range(len(trainingbaddatadir)):
        print(i)
        batch = trainingbaddatadir.__getitem__(i)
        print(batch)

    return
    testfile = "/media/turro/CHARLES/fiip/data/netfeeder/pretraindiscriminator/validation/bad/475badaudiocurated10.pkl"
    descriminator = keras.models.load_model('/home/turro/Repos/FIIP/pretraineddiscriminator.mdl')
    testdata = pickle.load(open(testfile,"rb"))
    testdata = np.expand_dims(testdata,axis = 0)
    testdata = np.expand_dims(testdata,axis = 3)
    print(testdata.shape)
    print(model.predict(testdata))
    generator = FIIPGenerator((freqidxcut,41,1))




def pretraingenerator():
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    targetsamplingrate = metadata["targetsamplerate"]
    chunkduration = float(metadata["netdT"])
    freqidxcut = metadata["freqidxcut"]

    Gtrainingbaddatadir   = os.path.join(metadata["basedir"],metadata["baddatalocations"][0][0])
    Gvalidationbaddatadir = os.path.join(metadata["basedir"],metadata["baddatalocations"][1][0])
    Dtraininggooddatadir = os.path.join(metadata["basedir"],metadata["gooddatalocations"][0][0])
    #trainingbaddatadir = os.path.join(metadata["basedir"],metadata["baddatalocations"][0][0])
    Dtrainingbaddatadir = os.path.join(metadata["basedir"],metadata["fakedir"])
    Dvalidationgooddatadir = os.path.join(metadata["basedir"],metadata["gooddatalocations"][1][0])
    Dvalidationbaddatadir = os.path.join(metadata["basedir"],metadata["baddatalocations"][1][0])

    discriminator = keras.models.load_model('/home/turro/Repos/FIIP/pretraineddiscriminator.mdl')
    discriminator.trainable = False
    generator = FIIPGenerator((freqidxcut,41,1))
    generator.build((8,freqidxcut,41,1))

    GAN = tf.keras.models.Sequential()
    GAN.add(generator)
    GAN.add(discriminator)
    GAN.build((8,freqidxcut,41,1))

    GAN.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives()
        ]
    )
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode = "min",verbose = 1,patience=10,restore_best_weights=True)

    print(GAN.summary())
    print(generator.summary())
    for i in range(3):
        preTrainDataset = GeneratorFeeder(Gtrainingbaddatadir,(freqidxcut,41,1))
        preValidDataset = GeneratorFeeder(Gvalidationbaddatadir,(freqidxcut,41,1))
        hist = GAN.fit(
            x = preTrainDataset,
            epochs=1,
            verbose=1,
            validation_data=preValidDataset,
            callbacks= [es]
        )
        for i in range(len(preTrainDataset)):
            batch = preTrainDataset[i]
            generated = generator.predict(batch[0])
            for j in range(generated.shape[0]):
                with open(os.path.join(metadata["basedir"],metadata["fakedir"],"{}-{}.pkl".format(str(i),str(j))),"wb+") as fh:
                    pickle.dump(np.squeeze(generated[j]),fh)
        preTrainDiscDataset = PreTrainDiscrFeeder(Dtraininggooddatadir,Dtrainingbaddatadir,(freqidxcut,41,1))
        preValidDiscDataset = PreTrainDiscrFeeder(Dvalidationgooddatadir,Dvalidationbaddatadir,(freqidxcut,41,1))
        hist = discriminator.fit(
            x = preTrainDiscDataset,
            epochs=1,
            verbose=1,
            validation_data=preValidDiscDataset,
            callbacks= [es]
        )
def pretraindiscriminator():
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    targetsamplingrate = metadata["targetsamplerate"]
    freqidxcut = metadata["freqidxcut"]
    chunkduration = float(metadata["netdT"])

    traininggooddatadir = os.path.join(metadata["basedir"],metadata["gooddatalocations"][0][0])
    #trainingbaddatadir = os.path.join(metadata["basedir"],metadata["baddatalocations"][0][0])
    trainingbaddatadir = os.path.join(metadata["basedir"],metadata["fakedir"])
    validationgooddatadir = os.path.join(metadata["basedir"],metadata["gooddatalocations"][1][0])
    validationbaddatadir = os.path.join(metadata["basedir"],metadata["baddatalocations"][1][0])
    preTrainDiscDataset = PreTrainDiscrFeeder(traininggooddatadir,trainingbaddatadir,(freqidxcut,41,1))
    preValidDiscDataset = PreTrainDiscrFeeder(validationgooddatadir,validationbaddatadir,(freqidxcut,41,1))

    discriminator = FIIPDiscriminator((freqidxcut,41,1))
    print(discriminator.summary())
    discriminator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives()
        ]
    )

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode = "min",verbose = 1,patience=10,restore_best_weights=True)
    hist = discriminator.fit(
        x = preTrainDiscDataset,
        epochs=1,
        verbose=1,
        validation_data=preValidDiscDataset,
        callbacks= [es]
    )

    with open(os.path.join(metadata["basedir"],metadata["lasttraininghistory"]),"wb+") as fh:
        pickle.dump(hist.history,fh)
    discriminator.save("pretraineddiscriminator.mdl")
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
            "pretraingenerator": pretraingenerator,
            "sandbox":sandbox,
            "exit":exit
        }
        jobidx = []
        for i,j in enumerate(jobs):
            jobidx.append(jobs[j])
             
        while True:
            print("jobs")
            for i,j in enumerate(jobs):
                print(i, ": ", j)
            job = int(input("Select a job index: "))
            jobidx[job]()
