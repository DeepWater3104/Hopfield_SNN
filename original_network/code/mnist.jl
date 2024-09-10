using Base: @kwdef
using Parameters: @unpack
using Random
using Plots
using MLDatasets
include("network.jl")

Random.seed!(10)
T = 1000
dt = 1.0
num_patterns = 3
mnist = MNIST(:train).features
NX = size(mnist[:, :, 1], 2)
NY = size(mnist[:, :, 1], 1)
nt = UInt(T/dt)
t = Array{Float32}(1:nt)*dt

pattern = Vector{Matrix{Float32}}(undef, num_patterns)
pattern_vectorized = Vector{Vector{Float32}}(undef, num_patterns)
pattern_index = [rand(1:size(mnist, 3)) for i in 1:num_patterns]
for i=1:num_patterns
    pattern[i] = Matrix{Float32}(undef, NY, NX)
    pattern_vectorized[i] = Vector{Float32}(undef, NY*NX)
    pattern[i] = mnist[:, :, pattern_index[i]]
    pattern_vectorized[i] = vec(pattern[i])
end
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
savefig("../figure/mnist_typical_behavior.png")
