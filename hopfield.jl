using Base: @kwdef
using Parameters: @unpack
using Random
using Plots

@kwdef struct LIFParameter{FT}
    τ::FT = 20.
    wexec::FT = 5.
    vrest::FT = -65.
    θ::FT = -55.
    r_m::FT = 16.
    I_strength::FT = 1.5
    decay::FT = 0.5
end

@kwdef mutable struct LIF{FT}
    param::LIFParameter{FT} = LIFParameter{FT}()
    NX::UInt32
    NY::UInt32
    N::UInt32
    x::Vector{UInt32} = zeros(N)
    y::Vector{UInt32} = zeros(N)

    i_ext::Vector{FT} = zeros(N)

    v::Vector{FT} = fill(param.vrest, N)
    gsyn::Vector{FT} = zeros(N)
    spike::Vector{FT} = zeros(N)
    num_spikes::Vector{FT} = zeros(N)
end


function initialize!( variable::LIF, param::LIFParameter, pattern1::Vector, pattern2::Vector)
    @unpack NX, NY, N, x, y, i_ext, gsyn = variable
    @unpack wexec, I_strength = param

    for i=1:N
        y[i] = UInt32(floor((i-1) / NY)) + 1
        x[i] = UInt32(i - (y[i]-1)*NY)
    end

    # re-consider this input pattern later
    for i=1:N
        if x[i] < NX / 2
          i_ext[i] = pattern1[i]*I_strength
        end
    end
end


function update_LIF!( variable::LIF, param::LIFParameter, dt, time, SpikeTime::Vector, SpikeNeuron::Vector )
    @unpack v, gsyn, i_ext, spike, num_spikes, N = variable
    @unpack τ, wexec, vrest, θ, r_m, decay = param

    for i=1:N
        v[i] += dt*(vrest - v[i] + gsyn[i] + r_m * (i_ext[i] + rand() ) ) / τ
        if v[i] > θ
            spike[i] = 1
            v[i] = vrest
            push!(SpikeTime, time)
            push!(SpikeNeuron, i)
            num_spikes[i] += 1
        else
            spike[i] = 0
            v[i] = v[i]
        end
    end
end


function calculate_synaptic_current!( variable::LIF, param::LIFParameter, connection::Matrix, weight::Matrix )
    @unpack gsyn, N, spike = variable
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
    gsyn = gsyn * decay + r
end

function output_trace!( t, varr, variable::LIF )
    @unpack v, N = variable
    for i=1:N
        varr[i, t] = v[i]
    end
end

Random.seed!(10)
T = 1000
dt = 1.0
NX = 5
NY = 5
nt = UInt(T/dt)
t = Array{Float32}(1:nt)*dt

pattern1 = [1, 0, 0, 0, 1, 
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0, 
            1, 0, 0, 0, 1]

pattern2 = [0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0]

neurons = LIF{Float32}(N=NX*NY, NX=NX, NY=NY)
initialize!(neurons, neurons.param, pattern1, pattern2)

tmp = zeros(neurons.NX, neurons.NY, neurons.NX, neurons.NY)
max_connections = 0
for i=1:neurons.N
    num_connections = 0
    for j=1:neurons.N
        tmp[neurons.x[i], neurons.y[i], neurons.x[j], neurons.y[j]] += neurons.param.wexec * pattern1[i] * pattern1[j]
        tmp[neurons.x[i], neurons.y[i], neurons.x[j], neurons.y[j]] += neurons.param.wexec * pattern2[i] * pattern2[j]
        if 1e-10 < tmp[neurons.x[i], neurons.y[i], neurons.x[j], neurons.y[j]]
            num_connections += 1
        end
    end

    if max_connections < num_connections
        global max_connections = num_connections
    end
end

# store synaptic connection and its weight by ELL matrix
connection = zeros(UInt32, neurons.N, max_connections)
weight = zeros(neurons.N, max_connections)

for i=1:neurons.N
    num_connections = 0
    for j=1:neurons.N
        if 1e-10 < tmp[neurons.x[i], neurons.y[i], neurons.x[j], neurons.y[j]]
            num_connections += 1
            connection[i, num_connections] = (UInt32)(j)
            weight[i, num_connections] = tmp[neurons.x[i], neurons.x[i], neurons.x[j], neurons.x[j]]

        end
    end
end


# for recording
SpikeTime = []
SpikeNeuron = []
varr = zeros(neurons.N, nt)

# simulation
@time for i=1:nt
    update_LIF!(neurons, neurons.param, dt, t[i], SpikeTime, SpikeNeuron)
    output_trace!(i, varr, neurons)
    calculate_synaptic_current!(neurons, neurons.param, connection, weight)
end

# for outputting firing rate or each neurons
firing_rate = zeros(NX, NY)
for i=1:neurons.N
    firing_rate[neurons.y[i], neurons.x[i]] = neurons.num_spikes[i]
end
