import { createContext, useCallback, useContext, useState, type FormEvent, type ReactNode } from 'react';
import { useLocation } from 'react-router';
import { executeEchoCommand, type EchoCommandResult } from './echo-command';
import { resolveLabEchoContext } from './lab-echo-context';

interface EchoCommandContextValue {
  labLabel: string;
  input: string;
  setInput: (value: string) => void;
  busy: boolean;
  last: EchoCommandResult | null;
  expanded: boolean;
  setExpanded: (v: boolean) => void;
  submit: (e?: FormEvent) => void;
}

const EchoCommandContext = createContext<EchoCommandContextValue | null>(null);

export function EchoCommandProvider({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  const labContext = resolveLabEchoContext(pathname);
  const labLabel =
    labContext?.labName ??
    (pathname === '/echo' ? 'Echo Control Center' : 'QuLab');

  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [last, setLast] = useState<EchoCommandResult | null>(null);
  const [expanded, setExpanded] = useState(false);

  const submit = useCallback(
    async (e?: FormEvent) => {
      e?.preventDefault();
      const text = input.trim();
      if (!text || busy) return;

      setBusy(true);
      const outcome = await executeEchoCommand(text, { context: labContext });
      setLast(outcome);
      setExpanded(true);
      setBusy(false);
      if (outcome.ok) setInput('');
    },
    [input, busy, labContext]
  );

  return (
    <EchoCommandContext.Provider
      value={{
        labLabel,
        input,
        setInput,
        busy,
        last,
        expanded,
        setExpanded,
        submit,
      }}
    >
      {children}
    </EchoCommandContext.Provider>
  );
}

export function useEchoLabCommand(): EchoCommandContextValue {
  const ctx = useContext(EchoCommandContext);
  if (!ctx) {
    throw new Error('useEchoLabCommand must be used within EchoCommandProvider');
  }
  return ctx;
}
