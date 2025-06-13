import weight_normalise.Receptors as Receptors
import weight_normalise.Network as Network


initial_Vm = 1.3458754117369027
all_synapses = []
def create_synapse(send_neuron, receive_neuron, type):
    
    # create receptors accordingly
    if type == "AMPA":
        # temporal solution for weight randomise
        # Receptors.LigandGatedChannelFactory.set_params()
        # gMax, gP, rE, Vm, e, u_se, g_decay, g_rise, w, tau_rec, tau_pre, tau_post, tau_decay, tau_rise, 
        # learning_rate, label
        ampa_receptor = Receptors.AMPA(0.072, 1, 70, 1.35, 1, 0.8, 1, 1, 12, 10, 20, 10, 8, 10, 0.7, "AMPA")
        synapse = Network.Synapse(0.05, 0, send_neuron, receive_neuron, ampa_receptor)
        
    elif type == "AMPA+NMDA":
        # Receptors.LigandGatedChannelFactory.set_params()
        ampa_receptor = Receptors.AMPA(0.072, 1, 70, 1.35, 0.9, 1, 1, 1, 12, 10, 20, 10, 35, 7, 0.7, "AMPA")
        nmda_receptor = Receptors.NMDA(0.0012, 1, 70, 1.35, 0.9, 1, 1, 1, 13, 10, 20, 10, 15, 7, 0.7, "NMDA")
        synapse = Network.Synapse(0.05, 0, send_neuron, receive_neuron, ampa_receptor, nmda_receptor)
    
    elif type == "GABA":
        # Receptors.LigandGatedChannelFactory.set_params()
        # print(Receptors.LigandGatedChannelFactory.w_init_GABA)
        gaba_receptor = Receptors.GABA(0.004, 1, 140, 1.35, 0.9, 1, 1, 1, 12, 10, 20, 10, 20, 7, 0.7, "GABA")
        synapse = Network.Synapse(0.05, 0, send_neuron, receive_neuron, gaba_receptor)

    send_neuron.outgoing_synapses.append(synapse)
    receive_neuron.incoming_synapses.append(synapse)

    all_synapses.append(synapse)

def update_synapse_initial_values(infer_params):
    for synapse in all_synapses:
        for receptor in synapse.receptors:
            receptor.Vm = initial_Vm
            receptor.gP = 1
            
            receptor.e = infer_params["e"]
            receptor.u_se = infer_params["u_se"]
            receptor.g_decay = infer_params["g_decay"]
            receptor.g_rise = infer_params["g_rise"]
            receptor.w = infer_params["w"]
            receptor.tau_rec = infer_params["tau_rec"]
            receptor.tau_pre = infer_params["tau_pre"]
            receptor.tau_post = infer_params["tau_post"]

            if receptor.label == "GABA":
                receptor.gMax = infer_params["gMax_GABA"]
                receptor.tau_decay = infer_params["tau_decay_GABA"]
                receptor.tau_rise = infer_params["tau_rise_GABA"]
            
            elif receptor.label == "NMDA":
                receptor.tau_decay = infer_params["tau_decay_NMDA"]
                receptor.tau_rise = infer_params["tau_rise_NMDA"]
            
            elif receptor.label == "AMPA":
                receptor.tau_decay = infer_params["tau_decay_AMPA"]
                receptor.tau_rise = infer_params["tau_rise_AMPA"]