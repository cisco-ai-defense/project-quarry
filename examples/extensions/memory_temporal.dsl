rule "Confirmed memory write immediately after tool A"
condition:
    (any write in memory.write_committed where write.payload.source_tool == "tool_a")
    right_after
    called_tool("tool_a")
