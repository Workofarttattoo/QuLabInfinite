import { useLocation } from 'react-router';
import { shouldShowEchoCommandInline } from '../../lib/lab-echo-context';
import { useEchoLabCommand } from '../../lib/use-echo-lab-command';
import { EchoCommandPanel } from './EchoCommandPanel';

/** In-page Echo command — scrolls with content; shares state with the bottom dock. */
export function EchoLabCommandInline({ className = '' }: { className?: string }) {
  const { pathname } = useLocation();
  const visible = shouldShowEchoCommandInline(pathname);
  const echo = useEchoLabCommand();

  if (!visible) return null;

  return (
    <section className={`${className}`} aria-label="Echo instruction panel">
      <EchoCommandPanel
        variant="inline"
        labLabel={echo.labLabel}
        input={echo.input}
        busy={echo.busy}
        last={echo.last}
        expanded={echo.expanded}
        onInputChange={echo.setInput}
        onSubmit={echo.submit}
        onDismissResponse={() => echo.setExpanded(false)}
      />
    </section>
  );
}
