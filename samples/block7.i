loadI 1024 => r0    // load base address
        loadI 1 => r1       // a = 1
        loadI 2 => r2       // b = 2
        subI r2, 4 => r3    // c = b - 4
        add r1, r2 => r4    // d = a + b
        addI r4, 1 => r5    // e = d + 1
        mult r3, r5 => r6   // f = e - c * e (c*e calculation)
        sub r5, r6 => r6    // f - e - (c*e) (subtraction calculation)
        add r4, r5 => r7    // g = d + e
        add r6, r7 => r7    // g = d + e + f
        add r7, r1 => r8    // h = g + a
        storeAI r8 => r0, 0 // store h in memory
        outputAI r0, 0      // print output