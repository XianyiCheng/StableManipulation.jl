function is_same_mode(m1, m2)
    n = length(m1)
    n2 = length(m2)
    if n != n2
        return false
    end
    for i = 1:n
        if m1[i] != m2[i]
            return false
        end
    end
    return true
end
