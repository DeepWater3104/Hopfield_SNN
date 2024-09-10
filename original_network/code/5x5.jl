using Base: @kwdef
using Parameters: @unpack
using Random
using Plots
using MLDatasets
include("network.jl")

Random.seed!(10)
T = 1000
dt = 1.0
nt = UInt(T/dt)
t = Array{Float32}(1:nt)*dt
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
pattern = Vector{Matrix{Float32}}(undef, num_patterns)
pattern_vectorized = Vector{Vector{Float32}}(undef, num_patterns)
pattern[1] = Matrix{Float32}(undef, NY, NX)
pattern_vectorized[1] = Vector{Float32}(undef, NY*NX)
pattern[1] = pattern1
pattern_vectorized[1] = vec(pattern[1])
pattern[2] = Matrix{Float32}(undef, NY, NX)
pattern_vectorized[1] = Vector{Float32}(undef, NY*NX)
pattern[2] = pattern2
pattern_vectorized[2] = vec(pattern[2])

recall_pattern = 2

neurons = LIF{Float32}(N=NX*NY, NX=NX, NY=NY)
initialize!(neurons, neurons.param, pattern_vectorized, recall_pattern)

# for recording
SpikeTime = []
SpikeNeuron = []
varr = zeros(nt, neurons.N)

# simulation
@time for i=1:nt
    update_LIF!(neurons, neurons.param, dt, t[i], SpikeTime, SpikeNeuron)
    output_trace!(i, varr, neurons)
    calculate_synaptic_current!(neurons, neurons.param)
end

# output firing rate of each neurons
firing_rate = zeros(NY, NX)
for i=1:neurons.N
    firing_rate[neurons.y[i], neurons.x[i]] = neurons.num_spikes[i]/(T*0.001)
end

p1 = heatmap(transpose(firing_rate), yflip=true, title="output")
p2 = heatmap(transpose(pattern[1]), yflip=true, title="embedded pattern1")
p3 = heatmap(transpose(pattern[2]), yflip=true, title="embedded pattern2")
plot(p1, p2, p3, layout=(3,1))
#savefig("../figure/5x5_typical_behavior.png")
