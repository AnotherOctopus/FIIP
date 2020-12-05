
import soundfile as sf
from scipy.fft import fft
import json
import os
import pickle

datajson = "/home/turro/Repos/FIIP/processingmetadata.json"
with open(datajson) as jsonfh:
    metadata = json.load(jsonfh)

def rawtochunks(indir,outdir,chunkduration):
    rawwavs = []
    for filename in os.listdir(indir):
        if filename.endswith((".wav",".WAV")):
            rawwavs.append(os.path.join(indir,filename))
    idx = 0
    for wav in rawwavs:
        rawdata,samplingrate = sf.read(wav)
        chunks = []
        chunkstep = int(chunkduration*samplingrate)
        for chunkidx in range(0,len(rawdata),chunkstep):
            for band in range(rawdata.shape[1]):
                chunks.append(rawdata[chunkidx : chunkidx + chunkstep,band]) #[band])
        for chunk in chunks:
            with open(os.path.join(outdir,"{}.pkl".format(str(idx).zfill(6))),"wb+") as fh:
                pickle.dump(chunk,fh)
            idx += 1

def chunktofreq(indir,outdir):
    for filename in os.listdir(indir):
        with open(os.path.join(indir,filename),"rb")  as fh:
            chunk = pickle.load(fh)
        with open(os.path.join(outdir,filename),"wb+") as fh:
            freq = fft(chunk)
            pickle.dump(freq,fh)

def badsoundtochunks():
    chunkduration = metadata["netdT"]
    badsounddir = os.path.join(metadata["basedir"],metadata["badfreqdir"])
    badsoundtimechunkdir = os.path.join(metadata["basedir"],metadata["badactortimechunks"])
    for actor in metadata["actors"]:
        rawtochunks(os.path.join(badsounddir,actor),os.path.join(badsoundtimechunkdir,actor),chunkduration)

def generalrawtochunks():
    chunkduration = metadata["netdT"]
    generalcorpusdir = os.path.join(metadata["basedir"],metadata["rawcorpusesdir"])
    generaltimechunkdir = os.path.join(metadata["basedir"],metadata["generaltimechunks"])
    rawtochunks(generalcorpusdir,generaltimechunkdir,chunkduration)


def generalchunkstofreq():
    generaltimechunkdir = os.path.join(metadata["basedir"],metadata["generaltimechunks"])
    generalfreqchunkdir = os.path.join(metadata["basedir"],metadata["generalcurated"])
    chunktofreq(generaltimechunkdir,generalfreqchunkdir)

if __name__ == "__main__":
        jobs = {
            "generalrawtochunks":generalrawtochunks,
            "generalchunkstofreq":generalchunkstofreq,
            "badrawtochunks":badsoundtochunks,
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
