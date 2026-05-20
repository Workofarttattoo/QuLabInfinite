import type { FormEvent } from 'react';
import type { EchoCommandResult } from '../../lib/echo-command';

export interface EchoCommandPanelProps {
  labLabel: string;
  input: string;
  busy: boolean;
  last: EchoCommandResult | null;
  expanded: boolean;
  onInputChange: (value: string) => void;
  onSubmit: (e?: FormEvent) => void;
  onDismissResponse?: () => void;
  /** inline = in page flow; docked = fixed bottom bar */
  variant?: 'inline' | 'docked';
}

export function EchoCommandPanel({
  labLabel,
  input,
  busy,
  last,
  expanded,
  onInputChange,
  onSubmit,
  onDismissResponse,
  variant = 'docked',
}: EchoCommandPanelProps) {
  const wrapperClass =
    variant === 'inline'
      ? 'w-full'
      : 'pointer-events-auto mx-auto w-full max-w-[1200px] px-4 md:px-8';

  return (
    <div className={wrapperClass}>
      {last && expanded && (
        <div
          className={`mb-2 glass-panel rounded-lg border px-4 py-3 text-sm ${
            last.ok ? 'border-[#00dbe9]/40' : 'border-[#ffb4ab]/50'
          }`}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <p className={`font-semibold ${last.ok ? 'text-[#00dbe9]' : 'text-[#ffb4ab]'}`}>
                {last.summary}
              </p>
              {last.detail && (
                <p className="mt-1 text-[#b9cacb] text-xs whitespace-pre-wrap break-words">{last.detail}</p>
              )}
              {last.tool && (
                <p className="mt-1 font-mono text-[10px] text-[#849495] uppercase tracking-wider">
                  via {last.tool}
                </p>
              )}
            </div>
            {onDismissResponse && (
              <button
                type="button"
                onClick={onDismissResponse}
                className="text-[#849495] hover:text-white shrink-0"
                aria-label="Dismiss response"
              >
                <span className="material-symbols-outlined text-lg">close</span>
              </button>
            )}
          </div>
        </div>
      )}

      <form
        onSubmit={onSubmit}
        className="glass-panel rounded-xl border-2 border-[#00dbe9]/60 shadow-[0_0_32px_rgba(0,219,233,0.15)] overflow-hidden"
      >
        <div className="flex items-center gap-2 px-3 py-2 border-b border-white/10 bg-[#00dbe9]/15">
          <span className="material-symbols-outlined text-[#00dbe9] text-xl echo-icon-filled">
            psychology
          </span>
          <div className="flex-1 min-w-0">
            <p className="text-[10px] font-bold tracking-[0.2em] text-[#00dbe9] uppercase">
              Instruct Echo
            </p>
            <p className="text-[11px] text-[#b9cacb] truncate">
              Onboard AI · {labLabel}
              <span className="hidden sm:inline"> · try: analyze graphene, status, help</span>
            </p>
          </div>
        </div>

        <div className="flex items-stretch">
          <input
            type="text"
            value={input}
            onChange={(e) => onInputChange(e.target.value)}
            disabled={busy}
            autoComplete="off"
            spellCheck={false}
            className="flex-1 min-w-0 min-h-[52px] bg-[#0a0e14] border-0 px-4 py-3 text-[16px] text-[#dbfcff] placeholder:text-[#849495] focus:outline-none focus:ring-2 focus:ring-inset focus:ring-[#00dbe9]/50"
            placeholder={`Type a command for Echo in ${labLabel}…`}
            aria-label={`Instruction for Echo in ${labLabel}`}
          />
          <button
            type="submit"
            disabled={busy || !input.trim()}
            className="px-5 md:px-8 min-h-[52px] bg-[#00dbe9] text-[#00363a] text-xs font-bold uppercase tracking-[0.15em] hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
          >
            {busy ? '…' : 'Send'}
          </button>
        </div>
      </form>
    </div>
  );
}
