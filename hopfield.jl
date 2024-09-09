using Base: @kwdef
using Parameters: @unpack
using Random
using Plots
using MLDatasets

@kwdef struct LIFParameter{FT}
    τ::FT = 20.
    wexec::FT = 2.
    vrest::FT = -65.
    θ::FT = -55.
    r_m::FT = 16.
    I_strength::FT = 5.
    decay::FT = 0.5
    vpeak::FT = 20.
end

@kwdef mutable struct LIF{FT}
    param::LIFParameter{FT} = LIFParameter{FT}()
    NX::UInt32
    NY::UInt32
    N::UInt32
    x::Vector{UInt32} = zeros(N)
    y::Vector{UInt32} = zeros(N)

    connection::Vector{Vector{UInt32}} = [Vector{UInt32}(undef, 0) for _ in 1:N]
    weight::Vector{Vector{Float64}} = [Vector{UInt32}(undef, 0) for _ in 1:N]

    i_ext::Vector{FT} = zeros(N)

    v::Vector{FT} = fill(param.vrest, N)
    gsyn::Vector{FT} = zeros(N)
    spike::Vector{FT} = zeros(N)
    num_spikes::Vector{FT} = zeros(N)
end


function initialize!( variable::LIF, param::LIFParameter, pattern::Vector, recall_pattern)
    @unpack NX, NY, N, x, y, i_ext, gsyn, connection, weight = variable
    @unpack wexec, I_strength = param

    for i=1:N
        x[i] = UInt32(floor((i-1) / NY)) + 1
        y[i] = UInt32(i - (x[i]-1)*NY)
    end

    tmp = zeros(neurons.N, neurons.N)
    num_connections = zeros(UInt32, N)
    for i=1:neurons.N
        for j=1:neurons.N
            for k=1:length(pattern)
                tmp[i, j] += neurons.param.wexec * pattern[k][i] * pattern[k][j] / length(pattern)
            end
            if 1e-10 < tmp[i, j]
                num_connections[i] += 1
            end
        end
    end

    # store synaptic connection and its weight with sparse ELL matrix format
    for i=1:N
        connection[i] = zeros(UInt32, num_connections[i])
        weight[i] = zeros(Float64, num_connections[i])
    end

    for i=1:neurons.N
        num_connections = 0
        for j=1:neurons.N
            if 1e-10 < tmp[i, j]
                num_connections += 1
                connection[i][num_connections] = j
                weight[i][num_connections] = tmp[i, j]
            end
        end
    end

    for i=1:N
        if y[i] > NY/2
          #whcih to recall
          i_ext[i] = pattern[recall_pattern][i]*I_strength
        end
    end
end


function update_LIF!( variable::LIF, param::LIFParameter, dt, time, SpikeTime::Vector, SpikeNeuron::Vector )
    @unpack v, gsyn, i_ext, spike, num_spikes, N = variable
    @unpack τ, wexec, vrest, θ, r_m, decay, vpeak = param

    for i=1:N
        if spike[i] == 1
            v[i] = vrest
        end
        if v[i] > θ
            spike[i] = 1
            v[i] = vpeak
            push!(SpikeTime, time)
            push!(SpikeNeuron, i)
            num_spikes[i] += 1
        else
            spike[i] = 0
            v[i] += dt*(vrest - v[i] + gsyn[i] + r_m * (i_ext[i] + rand() ) ) / τ
        end
    end
end


function calculate_synaptic_current!( variable::LIF, param::LIFParameter )
    @unpack gsyn, N, spike, connection, weight = variable
    @unpack decay = param

    r = zeros(N)
    for pre = 1:N
        if spike[pre] == 1
            # loop about all the post-synaptic neuron
            for i=1:size(connection, 2)
                if connection[pre, i] != 0
                    r[connection[pre, i]] += 1*weight[pre, i]
                end
            end
        end
    end
    for pst = 1:N
        gsyn[pst] = gsyn[pst] * decay + r[pst]
    end
end


function output_trace!( t, varr, variable::LIF )
    @unpack v, N = variable
    for i=1:N
        varr[t, i] = v[i]
    end
end

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
savefig("typical_behavior.png")
