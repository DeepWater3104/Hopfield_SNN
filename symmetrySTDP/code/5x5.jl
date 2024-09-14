using Base: @kwdef
using Parameters: @unpack
using Random
using Plots
using MLDatasets
include("network.jl")

Random.seed!(10)
rng = MersenneTwister(1234)
num_patterns = 2
pattern1 = [1 0 0 0 1;
            0 1 0 1 0;
            0 0 1 0 0;
            0 1 0 1 0;
            1 0 0 0 1]
pattern2 = [0 0 1 0 0;
            0 0 1 0 0;
            1 1 1 1 1;
            0 0 1 0 0;
            0 0 1 0 0]
NX = size(pattern1, 2)
NY = size(pattern1, 1)
N = NX*NY
pattern = Vector{Matrix{Float32}}(undef, num_patterns)
pattern_vectorized = Vector{Vector{Float32}}(undef, num_patterns)
pattern[1] = Matrix{Float32}(undef, NY, NX)
pattern[2] = Matrix{Float32}(undef, NY, NX)
pattern_vectorized[1] = Vector{Float32}(undef, NY*NX)
pattern_vectorized[2] = Vector{Float32}(undef, NY*NX)
pattern[1] = pattern1
pattern[2] = pattern2
pattern_vectorized[1] = vec(pattern[1])
pattern_vectorized[2] = vec(pattern[2])

T_per_pattern_learn =  1500
T_per_pattern_recall = 300
dt = 1e-2
dt_sampling = 1e-1
nt_per_pattern_learn = UInt(floor(T_per_pattern_learn/dt))
nt_per_pattern_recall = UInt(floor(T_per_pattern_recall/dt))
nt_per_pattern_learn_sampling = UInt(floor(T_per_pattern_learn/dt_sampling))
nt_per_pattern_recall_sampling= UInt(floor(T_per_pattern_recall/dt_sampling))
#time = Array{Float32}(1:(nt_per_pattern_learn_sampling+nt_per_pattern_recall_sampling)*num_patterns)*dt_sampling
time = [i*dt_sampling for i=1:(nt_per_pattern_learn_sampling+nt_per_pattern_recall_sampling)*num_patterns]

neurons = LIF{Float32}(N=NX*NY, NX=NX, NY=NY, wexec=1.0)
initialize!(neurons, neurons.param)

# for recording
SpikeTime = []
SpikeNeuron = []
num_spikes_learn = zeros(num_patterns, N)
num_spikes_recall = zeros(num_patterns, N)
varr = zeros((nt_per_pattern_learn_sampling+nt_per_pattern_recall_sampling)*num_patterns, neurons.N )
weightarr = zeros( (nt_per_pattern_learn_sampling+nt_per_pattern_recall_sampling)*num_patterns, neurons.N, neurons.N )

# simulation (learning phase)
sampling_index = 1
@time for i=1:num_patterns
    for j=1:N
        neurons.poisson[j] = (pattern_vectorized[i][j] > 0)
    end
    for j=(i-1)*nt_per_pattern_learn+1:i*nt_per_pattern_learn
        update_LIF!(neurons, neurons.param, dt, j*dt, SpikeTime, SpikeNeuron, rng)
        calculate_synaptic_current!(neurons, neurons.param, dt)
        stdp!(neurons, neurons.param, dt)
        if (time[sampling_index] - j*dt) < 1e-10
            output_trace!(sampling_index, varr, weightarr, neurons)
            global sampling_index += 1
        end
    end
    num_spikes_learn[i, :] = neurons.num_spikes
    neurons.num_spikes = zeros(N)
end

neurons.poisson = zeros(N)

# simulation (recall phase)
@time for i=1:num_patterns
    for j=1:N
        if neurons.y[j] < 3
            neurons.i_ext[j] = neurons.param.I_strength*(pattern_vectorized[i][j] > 0)
        end
    end
    for j=num_patterns*nt_per_pattern_learn+(i-1)*nt_per_pattern_recall+1:num_patterns*nt_per_pattern_learn+i*nt_per_pattern_recall
        update_LIF!(neurons, neurons.param, dt, j*dt, SpikeTime, SpikeNeuron, rng)
        calculate_synaptic_current!(neurons, neurons.param, dt)
        if (time[sampling_index] - j*dt) < 1e-10
            output_trace!(sampling_index, varr, weightarr, neurons)
            global sampling_index += 1
        end
    end
    num_spikes_recall[i, :] = neurons.num_spikes
    neurons.num_spikes = zeros(N)
end

# output firing rate of each neurons
num_spikes_learn_matrix  = zeros(2, NY, NX)
num_spikes_recall_matrix = zeros(2, NY, NX)
for p=1:num_patterns
    num_spikes_learn_matrix[p, :, :] = reshape(num_spikes_learn[p, :], NY, NX)
    num_spikes_recall_matrix[p, :, :] = reshape(num_spikes_recall[p, :], NY, NX)
end

p1 = heatmap(transpose(num_spikes_learn_matrix[1, :, :]), yflip=true, title="learn1", color=:Blues)
p2 = heatmap(transpose(num_spikes_learn_matrix[2, :, :]), yflip=true, title="learn2", color=:Blues)
p3 = heatmap(transpose(num_spikes_recall_matrix[1, :, :]), yflip=true, title="recall1", color=:Blues)
p4 = heatmap(transpose(num_spikes_recall_matrix[2, :, :]), yflip=true, title="recall2", color=:Blues)
p5 = heatmap(transpose(pattern[1]), yflip=true, title="embedded pattern1", color=:Blues)
p6 = heatmap(transpose(pattern[2]), yflip=true, title="embedded pattern2", color=:Blues)
plot(p1, p2, p3, p4, p5, p6, layout=(3, 2))
savefig("failure.png")

a = reshape(weightarr, size(weightarr, 1), size(weightarr, 2)^2)
plot(time, a)
savefig("synapse_growth.png")
