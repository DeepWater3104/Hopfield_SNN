using Base: @kwdef
using Parameters: @unpack
using Random
using Plots
using MLDatasets
include("network.jl")

Random.seed!(2)
#Random.seed!(10) # this networks is not good at storing 4 and 2 at the same time
rng = MersenneTwister(1234)
T = 1000
dt = 0.01
dt_sampling = 0.1
nt = UInt(T/dt)
nt_sampling = UInt(T/dt_sampling)
time = Array{Float32}(1:nt_sampling)*dt_sampling

num_patterns = 2
mnist = MNIST(:train).features
NX = size(mnist[:, :, 1], 2)
NY = size(mnist[:, :, 1], 1)









pattern = Vector{Matrix{Float32}}(undef, num_patterns)
pattern_vectorized = Vector{Vector{Float32}}(undef, num_patterns)

pattern_index = [rand(1:size(mnist, 3)) for i in 1:num_patterns]
for i=1:num_patterns
    pattern[i] = Matrix{Float32}(undef, NY, NX)
    pattern_vectorized[i] = Vector{Float32}(undef, NY*NX)
    pattern[i] = mnist[:, :, pattern_index[i]]
    pattern_vectorized[i] = vec(pattern[i])
end

recall_pattern = 1


neurons = LIF{Float32}(N=NX*NY, NX=NX, NY=NY, wexec=0.5)
initialize!(neurons, neurons.param, pattern_vectorized, recall_pattern)

# for recording
SpikeTime = []
SpikeNeuron = []
varr = zeros(nt_sampling, neurons.N)

# simulation
sampling_index = 1
@time for i=1:nt
    update_LIF!(neurons, neurons.param, dt, dt*i, SpikeTime, SpikeNeuron, rng)
    calculate_synaptic_current!(neurons, neurons.param, dt)
    if (time[sampling_index] - i*dt) < 1e-10
        output_trace!(sampling_index, varr, neurons)
        global sampling_index += 1
    end

end


num_spikes_matrix = reshape(neurons.num_spikes, NY, NX)
p1 = heatmap(transpose(num_spikes_matrix), yflip=true, title="recall pattern1", color=:Blues)
p2 = heatmap(transpose(pattern[1]), yflip=true, title="embedded pattern1", color=:Blues)
p3 = heatmap(transpose(pattern[2]), yflip=true, title="embedded pattern2", color=:Blues)
plot(p1, p2, p3, layout=(3,1), size=(400, 700))
savefig("../figure/mnist_typical_behavior.png")
