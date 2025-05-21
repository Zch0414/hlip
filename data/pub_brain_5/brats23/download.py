import synapseclient
import synapseutils


if __name__ == "__main__":
    target_path = '/data/brats2023/'
    syn = synapseclient.Synapse() 
    syn.login(authToken="authToken") # replace with your token
    files = synapseutils.syncFromSynapse(syn, 'syn51156910', path=target_path)