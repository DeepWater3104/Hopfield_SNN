@kwdef struct LIFParameter{FT}
    # membrane parameters
    τ::FT = 20.
    vrest::FT = -65.
    θ::FT = -55.
    r_m::FT = 16.
    vpeak::FT = 20.

    # synaptic parameters
    decay::FT = 0.5
    wexec::FT = 2.
    A_p::FT = 0.01
    A_m::FT = A_p
    #τ_p::FT = 2e-1
    #τ_m::FT = 2e-1
    τ_p::FT = 2e-1
    τ_m::FT = 2e-1

    # poisson random
    poisson_rate::FT = 200. # Hz
    poisson_weight::FT = 10.

    # others
    I_strength::FT = 5.
end

@kwdef mutable struct LIF{FT}
    param::LIFParameter{FT} = LIFParameter{FT}()

    # network parameters
    NX::UInt32
    NY::UInt32
    N::UInt32
    x::Vector{UInt32} = zeros(N)
    y::Vector{UInt32} = zeros(N)


    # state variables
    v::Vector{FT} = fill(param.vrest, N)
    gsyn::Vector{FT} = zeros(N)
    spike::Vector{FT} = zeros(N)
    x_pre::Vector{FT} = zeros(N)
    x_post::Vector{FT} = zeros(N)
    weight::Matrix{FT} = zeros(N, N)
    i_ext::Vector{FT} = zeros(N)
    poisson::Vector{Bool} = zeros(N)

    # others
    num_spikes::Vector{FT} = zeros(N)
end


function initialize!( variable::LIF, param::LIFParameter)
    @unpack NX, NY, N, x, y, i_ext, gsyn, weight = variable
    @unpack wexec, I_strength = param

    for i=1:N
        x[i] = UInt32(floor((i-1) / NY)) + 1
        y[i] = UInt32(i - (x[i]-1)*NY)
    end

    # re-consider the initialize value of synaptic weight
    # maybe should set the upper bound lower
    #weight .= wexec*rand(N, N)
    weight = zeros(N, N)
    for i=1:N
        weight[i, i] = 0.
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


function calculate_synaptic_current!( variable::LIF, param::LIFParameter, dt )
    @unpack gsyn, N, spike, weight, poisson = variable
    @unpack decay, poisson_rate, poisson_weight, wexec = param

    r = zeros(N)
    for pre = 1:N
        if spike[pre] == 1
            # loop about all the post-synaptic neuron
            for post=1:N
                r[post] += wexec*weight[pre, post]
            end
        end
    end 

    poisson_rand = rand(N)
    for pst = 1:N
        if poisson[pst] && (poisson_rand[pst] < poisson_rate*1e-3*dt)
            r[pst] += poisson_weight
        end
        gsyn[pst] = gsyn[pst] * decay + r[pst]
    end
end


function stdp!( variable::LIF, param::LIFParameter, dt )
    @unpack N, x_pre, x_post, spike, weight = variable
    @unpack A_m, A_p, τ_p, τ_m = param

    # update trace
    for i=1:N
        #x_pre[i] = x_pre[i]*(1-dt/τ_p) + spike[i]
        #x_post[i] = x_post[i]*(1-dt/τ_m) + spike[i]
        x_pre[i] = exp(-dt/τ_p)*x_pre[i] + spike[i]
        x_post[i] = exp(-dt/τ_m)*x_post[i] + spike[i]
    end

    dw = zeros(N, N)
    for pre=1:N
        for post=1:N
            if pre != post
                #dw[pre, post] = A_p * x_pre[pre]*spike[pre] - A_m*x_post[post]*spike[post]
                dw[pre, post] = A_p * x_pre[pre]*spike[pre] + A_m*x_post[post]*spike[post]
                weight[pre, post] += dw[pre, post]
            end
        end
    end
end

function output_trace!( t, varr, weightarr, variable::LIF )
    @unpack v, N, weight = variable
    for i=1:N
        varr[t, i] = v[i]
    end
    weightarr[t, :, :] .= weight
end
