import { createPortal } from 'react-dom';
import { useLocation } from 'react-router';
import { shouldShowEchoCommandBar } from '../../lib/lab-echo-context';
import { useEchoLabCommand } from '../../lib/use-echo-lab-command';
import { EchoCommandPanel } from './EchoCommandPanel';

/** Fixed bottom dock — portaled above lab chrome so the input is always clickable. */
export function EchoLabCommandBar() {
  const { pathname } = useLocation();
  const visible = shouldShowEchoCommandBar(pathname);
  const echo = useEchoLabCommand();

  if (!visible || typeof document === 'undefined') return null;

  return createPortal(
    <div
      className="fixed left-0 right-0 z-[9999] bottom-0 pb-[max(1rem,env(safe-area-inset-bottom))] pt-2 bg-gradient-to-t from-[#0a0e14] via-[#0a0e14]/95 to-transparent"
      role="region"
      aria-label="Echo lab command interface"
    >
      <EchoCommandPanel
        variant="docked"
        labLabel={echo.labLabel}
        input={echo.input}
        busy={echo.busy}
        last={echo.last}
        expanded={echo.expanded}
        onInputChange={echo.setInput}
        onSubmit={echo.submit}
        onDismissResponse={() => echo.setExpanded(false)}
      />
    </div>,
    document.body
  );
}
