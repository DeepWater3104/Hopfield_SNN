using Base: @kwdef
using Parameters: @unpack
using Random
using Plots
using MLDatasets

@kwdef struct LIFParameter{FT}
    τ::FT = 20.
    wexec::FT = 2.
    #wexec::FT = 0.
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


function initialize!( variable::LIF, param::LIFParameter, pattern1::Vector, pattern2::Vector)
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
            tmp[i, j] += neurons.param.wexec * pattern1[i] * pattern1[j]
            tmp[i, j] += neurons.param.wexec * pattern2[i] * pattern2[j]
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
        #if x[i] < 10
        #if x[i] < 10
        if x[i] < 5
          #whcih to recall
          i_ext[i] = pattern1[i]*I_strength
          #i_ext[i] = pattern2[i]*I_strength
        end
    end
end


function update_LIF!( variable::LIF, param::LIFParameter, dt, time, SpikeTime::Vector, SpikeNeuron::Vector )
    @unpack v, gsyn, i_ext, spike, num_spikes, N = variable
    @unpack τ, wexec, vrest, θ, r_m, decay, vpeak = param

    for i=1:N
        v[i] += dt*(vrest - v[i] + gsyn[i] + r_m * (i_ext[i] + rand() ) ) / τ
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
            v[i] = v[i]
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
pattern = MNIST(:train).features
NX = size(pattern[:, :, 1], 2)
NY = size(pattern[:, :, 1], 1)
nt = UInt(T/dt)
t = Array{Float32}(1:nt)*dt

pattern1 = pattern[:, :, 50]
pattern2 = pattern[:, :, 1111]
#for i=1:length(pattern[:, :,1], dims=1)
neurons = LIF{Float32}(N=NX*NY, NX=NX, NY=NY)
initialize!(neurons, neurons.param, vec(pattern1), vec(pattern2))

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
p2 = heatmap(transpose(pattern1), yflip=true, title="embedded pattern1")
p3 = heatmap(transpose(pattern2), yflip=true, title="embedded pattern2")
plot(p1, p2, p3, layout=(3,1))
savefig("typical_behavior.png")
