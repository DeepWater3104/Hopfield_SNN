@kwdef struct LIFParameter{FT}
    τ::FT = 20.
    I_strength::FT = 1.5
    vrest::FT = -65.
    θ::FT = -55.
    r_m::FT = 16.
    τ_syn::FT = 4.
end


@kwdef mutable struct LIF{FT}
    param::LIFParameter{FT} = LIFParameter{FT}()
    NX::UInt32
    NY::UInt32
    N::UInt32
    wexec::FT
    x::Vector{UInt32} = zeros(N)
    y::Vector{UInt32} = zeros(N)
    weight::Matrix{Float32} = zeros(N, N)
    i_ext::Vector{FT} = zeros(N)
    v::Vector{FT} = fill(param.vrest, N)
    gsyn::Vector{FT} = zeros(N)
    spike::Vector{FT} = zeros(N)
    num_spikes::Vector{FT} = zeros(N)
end


function initialize!( variable::LIF, param::LIFParameter, pattern::Vector, recall_pattern)
    @unpack NX, NY, N, x, y, i_ext, gsyn, weight, wexec = variable
    @unpack I_strength = param

    for i=1:N
        y[i] = UInt32(floor((i-1) / NY)) + 1
        x[i] = UInt32(i - (y[i]-1)*NY)
    end

    for pre=1:neurons.N
        for post=1:neurons.N
            if pre != post
                for k=1:length(pattern)
                    weight[pre, post] += (wexec * pattern[k][pre] * pattern[k][post]) / length(pattern)
                end
            end
        end
    end

    for i=1:N
        if y[i] > NY/2
            i_ext[i] = pattern[recall_pattern][i]*I_strength
        else
            i_ext[i] = 0
        end
    end
end


function update_LIF!( variable::LIF, param::LIFParameter, dt, time, SpikeTime::Vector, SpikeNeuron::Vector, rng )
    @unpack v, gsyn, i_ext, spike, num_spikes, N = variable
    @unpack τ, vrest, θ, r_m = param

    noise = (3*randn(rng, N).+1)*sqrt(dt)
    for i=1:N
        if v[i] > θ
            spike[i] = 1
            v[i] = vrest
            push!(SpikeTime, time)
            push!(SpikeNeuron, i)
            num_spikes[i] += 1
        else
            v[i] += ( (vrest - v[i] + gsyn[i] + r_m*i_ext[i])*dt + noise[i] ) / τ
            spike[i] = 0
        end
    end
end


function calculate_synaptic_current!( variable::LIF, param::LIFParameter, dt )
    @unpack gsyn, N, spike, weight = variable
    @unpack τ_syn= param

    r = zeros(N)
    for pre = 1:N
        if spike[pre] == 1
            for post = 1:N
                r[post] += weight[pre, post]
            end
        end
    end
    for post = 1:N
        gsyn[post] = gsyn[post] * exp(-dt/τ_syn) + r[post]
    end
end


function output_trace!( t, varr, variable::LIF )
    @unpack v, N = variable
    for i=1:N
        varr[t, i] = v[i]
    end
end
